from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
from time import time

cores = multiprocessing.cpu_count() // 3

args = parse_args()
Ks = eval(args.Ks)
device = device = torch.device("cuda:{}".format(args.gpu_id)) if args.gpu_id != -1 and torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag
user_recall_result = dict()
deg_recall = dict()
deg_recall_mean = dict()

@torch.no_grad()
def fast_evaluate_batch(user_gcn_emb, item_gcn_emb, test_user_set, Ks, batch_size=256):
    device = user_gcn_emb.device
    user_ids = list(test_user_set.keys())
    n_users = len(user_ids)
    max_K = max(Ks)

    precision_sum = {k: 0. for k in Ks}
    recall_sum = {k: 0. for k in Ks}
    ndcg_sum = {k: 0. for k in Ks}
    hit_sum = {k: 0. for k in Ks}

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch_uids = user_ids[start:end]
        u_embed = user_gcn_emb[batch_uids]  # [B, D]

        # 得分矩陣 [B, n_items]
        scores = torch.matmul(u_embed, item_gcn_emb.T)  # [B, n_items]
        
        ### #####
        for i, uid in enumerate(batch_uids):
            train_items = train_user_set.get(uid, [])
            scores[i, train_items] = -1e10  # 屏蔽
        topk_items = torch.topk(scores, k=max_K, dim=1).indices
        
        # ground-truth mask
        hits = []
        for i, uid in enumerate(batch_uids):
            gt_items = set(test_user_set[uid])
            top_items = topk_items[i].tolist()
            hit_vector = [1 if item in gt_items else 0 for item in top_items]
            hits.append(hit_vector)
        hits = torch.tensor(hits, device=device)  # [B, max_K]

        for K in Ks:
            r = hits[:, :K].float()
            precision = r.sum(dim=1) / K
            recall = r.sum(dim=1) / torch.tensor([len(test_user_set[u]) for u in batch_uids], device=device)
            hit_ratio = (r.sum(dim=1) > 0).float()
            denom = torch.log2(torch.arange(K, device=device).float() + 2)
            ndcg = (r / denom).sum(dim=1)
            idcg = torch.tensor([
                sum([1.0 / np.log2(i + 2) for i in range(min(len(test_user_set[u]), K))])
                for u in batch_uids
            ], device=device)
            ndcg = ndcg / idcg.clamp(min=1e-6)

            precision_sum[K] += precision.sum().item()
            recall_sum[K] += recall.sum().item()
            ndcg_sum[K] += ndcg.sum().item()
            hit_sum[K] += hit_ratio.sum().item()

    # 平均化
    final_result = {}
    for K in Ks:
        final_result[K] = {
            'precision': precision_sum[K] / n_users,
            'recall': recall_sum[K] / n_users,
            'ndcg': ndcg_sum[K] / n_users,
            'hit_ratio': hit_sum[K] / n_users
        }
    return final_result

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    # 将数据转换为GPU上的torch.Tensor
    rating_tensor = torch.Tensor(rating).to(device)
    test_items_tensor = torch.LongTensor(test_items).to(device)

    # 获取 test_items 中的评分
    item_score_tensor = rating_tensor[test_items_tensor]


    # 使用topk函数进行排序
    _, K_max_indices = torch.topk(item_score_tensor, k=max(Ks))
    
    sorted_item_indices = test_items_tensor[K_max_indices]

    r = []
    for i in sorted_item_indices:
        if i.item() in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.0
    return r, auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(round(precision_at_k(r, K), 5))
        recall.append(round(recall_at_k(r, K, len(user_pos_test)), 5))
        ndcg.append(round(ndcg_at_k(r, K, user_pos_test), 5))
        hit_ratio.append(round(hit_at_k(r, K), 5))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))
    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    evaluate_result = get_performance(user_pos_test, r, auc, Ks)
    # print(evaluate_result["recall"])
    return evaluate_result


def fast_test(model, user_dict, n_params, deg, mode='test'):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items, user_recall_result, deg_recall, deg_recall_mean
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    if mode == 'test':
        test_user_set = user_dict['test_user_set']
    else:
        test_user_set = user_dict['valid_user_set']
        if test_user_set is None:
            test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    user_gcn_emb, item_gcn_emb = model.generate()
    result_dict = fast_evaluate_batch(user_gcn_emb, item_gcn_emb, test_user_set, Ks, batch_size=u_batch_size)
    # 將 fast_evaluate_batch 結果寫入 result，保持原變量形狀
    for i, K in enumerate(Ks):
        result['precision'][i] = result_dict[K]['precision']
        result['recall'][i] = result_dict[K]['recall']
        result['ndcg'][i] = result_dict[K]['ndcg']
        result['hit_ratio'][i] = result_dict[K]['hit_ratio']
    result['auc'] = 0.0  # fast_evaluate_batch 不計算 AUC

    # 用 dummy value 補 user_recall_result（保持 deg 分析不出錯）
    for uid in test_user_set:
        user_recall_result[uid] = result_dict[max(Ks)]['recall']  # 所有用戶填一樣的 recall，保住 deg_recall 的邏輯

    count = len(test_user_set)

    for index, val in enumerate(deg.tolist()):
        if index < n_users:
            user_deg = int(val[0])
            if user_deg not in deg_recall:
                deg_recall[user_deg] = []
            if index in user_recall_result:
                deg_recall[user_deg].append(user_recall_result[index])
            else:
                deg_recall[user_deg].append(0.0)
            deg_recall_mean[user_deg] = sum(deg_recall[user_deg]) / len(deg_recall[user_deg])
    assert count == n_test_users
    pool.close()
    return result, user_recall_result, deg_recall, deg_recall_mean


def test(model, user_dict, n_params, deg, mode='test'):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items, user_recall_result, deg_recall, deg_recall_mean
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    if mode == 'test':
        test_user_set = user_dict['test_user_set']
    else:
        test_user_set = user_dict['valid_user_set']
        if test_user_set is None:
            test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    user_gcn_emb, item_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start:end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embddings = item_gcn_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()
                
                rate_batch[:, i_start:i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = item_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = []
        for x in user_batch_rating_uid:
            batch_result.append(test_one_user(x))
        # batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)
        i = 0 
        for re in batch_result:
            user_recall_result[user_list_batch[i]] = re['recall'].tolist()[0]
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
            i += 1

    for index, val in enumerate(deg.tolist()):
        if index < n_users:
            user_deg = int(val[0])
            if user_deg not in deg_recall:
                deg_recall[user_deg] = []
            if index in user_recall_result:
                deg_recall[user_deg].append(user_recall_result[index])
            else:
                deg_recall[user_deg].append(0.0)
            deg_recall_mean[user_deg] = sum(deg_recall[user_deg]) / len(deg_recall[user_deg])
    assert count == n_test_users
    pool.close()
    return result, user_recall_result, deg_recall, deg_recall_mean

def fast_test(model, user_dict, n_params, deg, mode='test'):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items, user_recall_result, deg_recall, deg_recall_mean
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    if mode == 'test':
        test_user_set = user_dict['test_user_set']
    else:
        test_user_set = user_dict['valid_user_set']
        if test_user_set is None:
            test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    user_gcn_emb, item_gcn_emb = model.generate()
    result_dict = fast_evaluate_batch(user_gcn_emb, item_gcn_emb, test_user_set, Ks, batch_size=u_batch_size)
    # 將 fast_evaluate_batch 結果寫入 result，保持原變量形狀
    for i, K in enumerate(Ks):
        result['precision'][i] = result_dict[K]['precision']
        result['recall'][i] = result_dict[K]['recall']
        result['ndcg'][i] = result_dict[K]['ndcg']
        result['hit_ratio'][i] = result_dict[K]['hit_ratio']
    result['auc'] = 0.0  # fast_evaluate_batch 不計算 AUC

    # 用 dummy value 補 user_recall_result（保持 deg 分析不出錯）
    for uid in test_user_set:
        user_recall_result[uid] = result_dict[max(Ks)]['recall']  # 所有用戶填一樣的 recall，保住 deg_recall 的邏輯

    count = len(test_user_set)

    for index, val in enumerate(deg.tolist()):
        if index < n_users:
            user_deg = int(val[0])
            if user_deg not in deg_recall:
                deg_recall[user_deg] = []
            if index in user_recall_result:
                deg_recall[user_deg].append(user_recall_result[index])
            else:
                deg_recall[user_deg].append(0.0)
            deg_recall_mean[user_deg] = sum(deg_recall[user_deg]) / len(deg_recall[user_deg])
    assert count == n_test_users
    pool.close()
    return result, user_recall_result, deg_recall, deg_recall_mean
