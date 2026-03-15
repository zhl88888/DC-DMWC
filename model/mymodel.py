import torch
from wavelet_transform import WaveletTransformModule
from feature_extractor import FeatureExtractor
from gcn_module import GCNModule
from transformer_module import TransformerModule
from segmentation_network import SegmentationNetwork
from discriminator import Discriminator
import torch.nn as nn

class GCNTransformerSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(GCNTransformerSegmentation, self).__init__()
        self.wavelet_transform = WaveletTransformModule(wavelet='db4', level=2)
        self.feature_extractor = FeatureExtractor()
        self.gcn_module = GCNModule(in_channels=2048, out_channels=512)
        self.transformer_module = TransformerModule(embed_dim=512, num_heads=8)
        self.segmentation_network = SegmentationNetwork(num_classes=num_classes)
        self.discriminator = Discriminator(num_classes=num_classes)

    def forward(self, x1, x2, adj_matrix):
        # 小波变换融合
        fused_image = self.wavelet_transform.forward(x1, x2)
        # 特征提取
        features = self.feature_extractor(fused_image)
        features = features.view(features.size(0), -1, features.size(1))
        # GCN模块
        gcn_features = self.gcn_module(features, adj_matrix)
        # Transformer模块
        transformer_features = self.transformer_module(gcn_features.unsqueeze(1)).squeeze(1)
        # 分割网络
        segmentation_output = self.segmentation_network(transformer_features)
        return segmentation_output

    def discriminate(self, x):
        return self.discriminator(x)