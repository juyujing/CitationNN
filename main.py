import os
import random

import torch
import numpy as np
from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable
from modules.MF import Matrix_Factorization
from utils.parser import parse_args

from utils.data_loader import load_data
from utils.evaluate import test, fast_test
from utils.helper import early_stopping
import optuna
import datetime
from modules.CitationNN import CitationNN
from modules.MCAP import MCAP
from modules.LightGCN import LightGCN
from modules.ApeGNN_APPNP import APPNP
from modules.ApeGNN_HT import HeatKernel

import os

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1, K=1, n_items=0):
    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user] :
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs * K)).to(device)
    return feed_dict


def opt_objective(trial, args, train_cf, user_dict, n_params, norm_mat, deg, outdeg, u2u, cite_graph):
    valid_res_list = []

    args.dim = trial.suggest_int('dim', 64, 512)
    args.l2 = trial.suggest_float('l2', 0, 1)
    args.context_hops = trial.suggest_int('context_hops', 1, 6)
    print(args)
    valid_best_result = main(args, args.seed, train_cf, user_dict, n_params, norm_mat, deg, outdeg, u2u, cite_graph)
    valid_res_list.append(valid_best_result)
    return np.mean(valid_res_list)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args, run, train_cf, user_dict, n_params, norm_mat, deg, outdeg, u2u, cite_graph):
    if args.fast_test:
        print('\nFast Test actived.\n')

    if args.dataset=='dblp':
        mode = 'user'
    else:
        mode = 'item'
    
    """define model"""
    if args.gnn == "CitationNN":
        model = CitationNN(n_params, args, norm_mat, u2u, cite_graph, mode).to(device)
    elif args.gnn == "MCAP":
        model = MCAP(n_params, args, norm_mat, u2u).to(device)
    elif args.gnn == "LightGCN":
        model = LightGCN(n_params, args, norm_mat).to(device)
    elif args.gnn == 'MF':
        model = Matrix_Factorization(n_params, args).to(device)
    elif args.gnn == 'ApeGNN_APPNP':
        model = APPNP(n_params, args, norm_mat, deg).to(device)
    elif args.gnn == 'ApeGNN_HT':
        model = HeatKernel(n_params, args, norm_mat, deg).to(device)
        
        
    """define optimizer"""
    optimizer = torch.optim.Adam([{'params': model.parameters(),
                                   'lr': args.lr}])
    n_items = n_params['n_items']
    cur_best_pre_0 = 0
    stopping_step = 0
    best_epoch = 0
    print("start training ...")

    hyper = {"dim": args.dim, "l2": args.l2, "hops": args.context_hops}
    print("Start hyper parameters: ", hyper)

    test_results = {}
    epoch2model = {}
    for epoch in range(args.epoch):
        
        if args.dataset=='dblp' and args.gnn=='CitationNN':
            # 硬編碼
            if (epoch==50 or epoch==80 or epoch==100 or epoch==120) or (epoch>120 and epoch%30==0):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']} at epoch {epoch}")
        
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  args.n_negs,
                                  args.K,
                                  n_items)
            batch_loss, _, _ = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size
        train_e_t = time()
        print('loss:', round(loss.item(), 2), "time: ", round(train_e_t - train_s_t, 2), 's')

        if epoch % args.step == 0:
            """testing"""
            train_res = PrettyTable()
            train_res.field_names = ["Phase", "Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg",
                                     "hit_ratio", "precision"]

            model.eval()
            test_s_t = time()
            if args.fast_test:
                print('\nFast Test actived. Please keep your attention on the screen at every second.\n')
                test_ret, user_result, deg_recall, deg_recall_mean = fast_test(model, user_dict, n_params, deg, mode='test')
            else:
                test_ret, user_result, deg_recall, deg_recall_mean = test(model, user_dict, n_params, deg, mode='test')
            test_e_t = time()

            train_res.add_row(
                ["Test", epoch, round(train_e_t - train_s_t, 2), round(test_e_t - test_s_t, 2), round(loss.item(), 2),
                 test_ret['recall'],
                 test_ret['ndcg'],
                 test_ret['hit_ratio'],
                 test_ret['precision']])
            test_results[epoch] = ["Test", epoch, round(train_e_t - train_s_t, 2), round(test_e_t - test_s_t, 2), round(loss.item(), 2),
                 test_ret['recall'],
                 test_ret['ndcg'],
                 test_ret['hit_ratio'],
                 test_ret['precision']]
            """valid"""
            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                if args.fast_test:
                    valid_ret, user_result, deg_recall, deg_recall_mean = fast_test(model, user_dict, n_params, deg, mode='valid')
                else:
                    valid_ret, user_result, deg_recall, deg_recall_mean = test(model, user_dict, n_params, deg, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    ["Valid", epoch, round(train_e_t - train_s_t, 2), round(test_e_t - test_s_t, 2), round(loss.item(), 2),
                     valid_ret['recall'],
                     valid_ret['ndcg'],
                     valid_ret['hit_ratio'],
                     valid_ret['precision']])
            print(train_res)
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.

            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][2], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=args.early_stop)
            if valid_ret['recall'][2] == cur_best_pre_0:
                best_epoch = epoch
            if should_stop:
                break
            epoch2model[epoch] = model

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f, best_epoch at %d' % (epoch, cur_best_pre_0, best_epoch))
    best_pretty = PrettyTable()
    best_pretty.field_names = ["Phase", "Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg",
                                     "hit_ratio", "precision"]
    best_pretty.add_row(test_results[best_epoch])
    print(best_pretty)
    """save weight"""
    if args.save:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(current_time)
        torch.save(epoch2model[best_epoch].state_dict(), args.out_dir + f'{args.dataset}_{args.dim}_{args.context_hops}_{args.l2}_{current_time}_' + args.gnn + '.ckpt')
    print("Seed:", run)
    print("End hyper parameters: ", hyper)
    print(f"Best valid_ret['recall']: ", cur_best_pre_0)
    return cur_best_pre_0


if __name__ == '__main__':
    """read args"""
    s = datetime.datetime.now()
    print("time of start: ", s)
    global args, device
    args = parse_args()
    set_seed(args.seed)
    if args.gpu_id != -1 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = torch.device("cpu")
    """build dataset"""
    train_cf, user_dict, n_params, norm_mat, deg, outdeg, u2u, cite_graph = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    trials = 1
    search_space = {'dim': [args.dim], 'context_hops': [args.context_hops], 'l2': [args.l2]}
    print("search_space: ", search_space)
    print("trials: ", trials)
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(lambda trial: opt_objective(trial, args, train_cf, user_dict, n_params, norm_mat, deg, outdeg, u2u, cite_graph), n_trials=trials)
    e = datetime.datetime.now()
    print(study.best_trial.params)
    print("time of end: ", e)
    print("finished all")
    
