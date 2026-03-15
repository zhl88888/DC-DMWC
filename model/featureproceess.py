import torch.nn as nn
import torch
from model.GCN import GCNModule
#from model.transformer import TransformerModule
#from utils.adjacency_matrix import AdjacencyMatrixBuilder
from utils.loss import SemanticConsistencyLoss, MMDLoss

class StructureAwareAttention(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(StructureAwareAttention, self).__init__()
        self.num_classes = num_classes
        self.class_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])

    def forward(self, x, labels):
        class_attn = torch.zeros_like(x)
        for c in range(self.num_classes):
            class_mask = (labels == c).float().unsqueeze(1).to('cuda')

            class_attn_c = self.class_attention[c](x)
            class_attn += class_mask * class_attn_c
        return class_attn


class FeatureInteractionModule(nn.Module):
    def __init__(self, in_channels, gcn_out_channels, transformer_embed_dim, num_heads):
        super(FeatureInteractionModule, self).__init__()
        self.gcn = GCNModule(in_channels, gcn_out_channels)
        self.transformer = TransformerModule(transformer_embed_dim, num_heads)
        self.mmd_loss = MMDLoss()
        self.upsample = nn.Upsample(size=(144, 144), mode='bilinear', align_corners=True)  # 添加上采样层

    def forward(self, src_fea, trg_fea, src_adj, trg_adj):
        batch_size, channels, height, width = src_fea.shape
        # GCN处理
        src_fea_gcn = self.gcn(src_fea, src_adj)
        trg_fea_gcn = self.gcn(trg_fea, trg_adj)

        # 重塑为特征图，形状为 [batch_size, gcn_out_channels, height, width]
        src_fea_gcn = src_fea_gcn.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        trg_fea_gcn = trg_fea_gcn.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        # 应用 Transformer 模块
        src_fea_transformer = self.transformer(src_fea_gcn.view(batch_size, -1, src_fea_gcn.shape[1]))
        src_fea_transformer = src_fea_transformer.view(batch_size, -1, height, width)
        trg_fea_transformer = self.transformer(trg_fea_gcn.view(batch_size, -1, trg_fea_gcn.shape[1]))
        trg_fea_transformer = trg_fea_transformer.view(batch_size, -1, height, width)

        #src_fea_trans=self.upsample(src_fea_transformer)
        #trg_fea_trans = self.upsample(trg_fea_transformer)
        # 特征对齐损失
        mmd_loss = self.mmd_loss( src_fea_transformer , trg_fea_transformer)

        # 特征交互（简单地将源域和目标域特征相加）
        interacted_fea =  src_fea_transformer + trg_fea_transformer

        return interacted_fea, mmd_loss