import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModule(nn.Module):
    """
    Transformer模块，用于捕捉特征之间的全局依赖关系。
    """

    def __init__(self, embed_dim, num_heads):
        super(TransformerModule, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 定义多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        # 定义前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 定义层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        前向传播函数。
        参数：
            x: 输入特征，形状为 [batch_size, channels, height, width]
        返回：
            输出特征，形状为 [batch_size, channels, height, width]
        """
        batch_size, channels, height, width = x.shape
        N = height * width

        # 展平特征图，形状为 [batch_size, N, channels]
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)

        # 调整输入形状以适应多头注意力机制，形状为 [N, batch_size, channels]
        x_flat = x_flat.permute(1, 0, 2)

        # 应用多头注意力机制
        attn_output, _ = self.multihead_attn(x_flat, x_flat, x_flat)

        # 残差连接和层归一化
        attn_output = self.norm1(attn_output + x_flat)

        # 应用前馈神经网络
        ffn_output = self.ffn(attn_output)

        # 残差连接和层归一化
        out = self.norm2(attn_output + ffn_output)

        # 重塑为特征图，形状为 [batch_size, channels, height, width]
        out = out.permute(1, 2, 0).view(batch_size, channels, height, width)

        return out

