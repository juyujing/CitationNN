import dgl
import torch
import numpy as np
from CitationNN import DGCN

def load_citation_data(file_path):
    """
    读取论文引用数据集，解析成 (src, dst) 形式
    :param file_path: 数据文件路径
    :return: (src_list, dst_list) -> (论文 A 引用 论文 B)
    """
    src_list = []
    dst_list = []

    with open(file_path, 'r') as f:
        for paper_id, line in enumerate(f.readlines()):
            tokens = list(map(int, line.strip().split()))
            num_citations = tokens[0]  # 第一列是引用数量
            if num_citations > 0:
                cited_papers = tokens[1:]  # 其余部分是被引用的论文
                for cited_paper in cited_papers:
                    src_list.append(paper_id)  # 论文 A (当前行的论文)
                    dst_list.append(cited_paper)  # 论文 B (被 A 引用)

    return src_list, dst_list


if __name__ == '__main__':
    # 加载数据
    citation_file = "citations.txt"  # 替换成你的文件路径
    src_nodes, dst_nodes = load_citation_data(citation_file)

    # 创建 DGL 异质图
    graph_data = {
        ('paper', 'cites', 'paper'): (src_nodes, dst_nodes),   # 论文 A 引用 论文 B
        ('paper', 'cited_by', 'paper'): (dst_nodes, src_nodes) # 论文 B 被 论文 A 引用 (反向边)
    }
    g = dgl.heterograph(graph_data)

    # 论文特征
    num_papers = g.num_nodes('paper')
    feature_dim = 128  # 论文特征维度
    try:
        paper_features = torch.load("paper_embeddings.pt")  # 读取预训练论文嵌入
        print("论文嵌入加载成功！")
    except FileNotFoundError:
        paper_features = torch.randn(num_papers, feature_dim)  # 随机初始化论文嵌入
        print("未找到论文嵌入，使用随机初始化")

    # 添加论文特征
    g.nodes['paper'].data['feat'] = paper_features

    # 打印图信息
    print(g)