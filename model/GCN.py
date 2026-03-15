import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNModule(nn.Module):
    """
    图卷积网络模块，用于捕捉特征之间的空间关系。
    """

    def __init__(self, in_channels, out_channels):
        super(GCNModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 定义图卷积层
        #self.gcn = nn.Linear(in_channels, out_channels)
        self.gcn = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),

            # 使用LayerNorm处理任意长度序列
            nn.LayerNorm(1024),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.LayerNorm(512),

            nn.Linear(512, out_channels)
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        """
        前向传播函数。
        参数：
            x: 输入特征，形状为 [batch_size, channels, height, width]
            edge_index: 邻接矩阵的边索引，形状为 [2, E]，其中 E 是边的数量
        返回：
            输出特征，形状为 [batch_size, out_channels, height, width]
        """
        batch_size, channels, height, width = x.shape
        N = height * width

        # 确保所有操作都在同一个设备上
        device = x.device

        # 展平特征图，形状为 [batch_size, N, channels]
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1).to(device)

        # 构建邻接矩阵，形状为 [batch_size, N, N]
        adj_matrix_list = []
        for i in range(batch_size):
            # 确保 edge_index 在正确的设备上
            edge_index_device = edge_index.to(device)
            adj_matrix = torch.sparse.FloatTensor(edge_index_device, torch.ones(edge_index_device.size(1), device=device), (N, N)).to_dense().to(device)
            adj_matrix_list.append(adj_matrix)
        adj_matrix = torch.stack(adj_matrix_list, dim=0).to(device)

        # 对邻接矩阵进行对称归一化
        adj_matrix = adj_matrix + torch.eye(N, device=device)  # 添加自连接
        degree_matrix = torch.sum(adj_matrix, dim=-1, keepdim=True)
        degree_matrix_inv_sqrt = torch.pow(degree_matrix, -0.5)
        degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0.0  # 处理无穷大值
        adj_matrix_norm = degree_matrix_inv_sqrt * adj_matrix * degree_matrix_inv_sqrt.permute(0, 2, 1)

        # 应用图卷积操作，形状为 [batch_size, N, out_channels]

        x_gcn = self.gcn(torch.matmul(adj_matrix_norm, x_flat))

        # 应用层归一化
        x_gcn = self.norm(x_gcn)

        # 重塑为特征图，形状为 [batch_size, out_channels, height, width]
        x_gcn = x_gcn.permute(0, 2, 1).view(batch_size, self.out_channels, height, width)

        return x_gcn
