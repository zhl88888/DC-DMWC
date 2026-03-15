import torch
import torch.nn.functional as F

def build_adjacency_matrix(features, k=3, threshold=0.5):
    """
    构建邻接矩阵。
    参数：
        features: 输入特征，形状为 [batch_size, channels, height, width]
        k: 每个节点的邻居数量
        threshold: 控制邻接矩阵稀疏性的阈值
    返回：
        邻接矩阵的边索引，形状为 [2, E]，其中 E 是边的数量
    """
    batch_size, channels, height, width = features.shape
    N = height * width

    # 展平特征图，形状为 [batch_size, channels, N]
    features_flat = features.view(batch_size, channels, -1)

    # 转置，形状为 [batch_size, N, channels]
    features_flat = features_flat.permute(0, 2, 1)

    # 计算特征之间的相似度
    similarity_matrix = torch.bmm(features_flat, features_flat.transpose(1, 2))
    similarity_matrix = similarity_matrix / (channels ** 0.5)  # 缩放

    # 归一化相似度矩阵
    similarity_matrix = F.softmax(similarity_matrix, dim=-1)

    # 构建邻接矩阵
    adj_matrix = torch.zeros((batch_size, N, N), device=features.device)
    for i in range(batch_size):
        _, indices = torch.topk(similarity_matrix[i], k, dim=1)
        adj_matrix[i].scatter_(1, indices, 1.0)

    # 应用阈值化
    adj_matrix[adj_matrix < threshold] = 0.0

    # 转换为边索引
    edge_index = []
    for i in range(batch_size):
        adj = adj_matrix[i]
        edge_index_batch = torch.nonzero(adj).t()
        edge_index.append(edge_index_batch)
    edge_index = torch.cat(edge_index, dim=1)

    return edge_index