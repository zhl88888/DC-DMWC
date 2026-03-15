import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


class PseudoLabelGenerator:
    def __init__(self,
                 num_classes,
                 feature_dim,
                 max_iters,
                 init_threshold=0.5,
                 momentum=0.999,
                 temperature=0.5,
                 device='cuda'):
        """
        Args:
            num_classes: 分割类别数
            feature_dim: 特征维度
            max_iters: 最大训练迭代次数
            init_threshold: 初始选择阈值
            momentum: 类别中心动量更新系数
            temperature: 温度缩放系数
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_iters = max_iters
        self.current_iter = 0
        self.momentum = momentum
        self.temperature = temperature
        self.device = device

        # 初始化类别中心 (使用K-means预训练或零初始化)
        self.class_centers = torch.zeros(num_classes, feature_dim).to(device)
        self.updated_counts = torch.zeros(num_classes).to(device)

        # 阈值参数
        self.base_threshold = init_threshold
        self.confidence_threshold = 0.8

        # 历史记录
        self.pseudo_quality_history = []

    def update_class_centers(self, features, pseudo_labels):
        """
        动量更新类别中心
        Args:
            features: 特征张量 (B, C, H, W)
            pseudo_labels: 伪标签 (B, H, W)
        """
        features = features.detach()
        pseudo_labels = F.interpolate(pseudo_labels.unsqueeze(1).float(),
                                      size=features.shape[-2:],
                                      mode='nearest').long().squeeze(1)

        # 展平特征和标签
        flat_features = features.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        flat_labels = pseudo_labels.reshape(-1)

        # 过滤低置信度区域
        with torch.no_grad():
            cos_sim = F.cosine_similarity(flat_features, self.class_centers[flat_labels], dim=1)
            valid_mask = cos_sim > self._get_current_threshold()

        # 按类别更新
        for cls_id in range(self.num_classes):
            cls_mask = (flat_labels == cls_id) & valid_mask
            if torch.sum(cls_mask) == 0:
                continue

            # 提取有效特征
            cls_features = flat_features[cls_mask]

            # 动量更新
            if self.updated_counts[cls_id] == 0:
                self.class_centers[cls_id] = cls_features.mean(dim=0)
            else:
                self.class_centers[cls_id] = (
                        self.momentum * self.class_centers[cls_id] +
                        (1 - self.momentum) * cls_features.mean(dim=0)
                )
            self.updated_counts[cls_id] += 1

    def generate_pseudo_labels(self, features, apply_crf=True):
        """
        生成伪标签
        Args:
            features: 特征张量 (B, C, H, W)
            apply_crf: 是否使用CRF后处理
        Returns:
            pseudo_labels: 伪标签 (B, H, W)
            confidence_mask: 高置信度区域掩码
        """
        B, C, H, W = features.shape

        # 计算余弦相似度（带温度缩放）
        norm_features = F.normalize(features, p=2, dim=1)
        cos_sim = torch.einsum('bchw,nc->bnhw', norm_features,
                               F.normalize(self.class_centers, p=2, dim=1)) / self.temperature

        # 动态阈值计算
        dynamic_threshold = self._get_current_threshold()

        # 生成候选伪标签
        max_sim, pseudo_labels = torch.max(cos_sim, dim=1)

        # 多维度筛选
        confidence_mask = self._create_confidence_mask(cos_sim, max_sim, dynamic_threshold)

        # 后处理
        if apply_crf:
            pseudo_labels = self._crf_postprocessing(features, pseudo_labels)

        # 记录质量指标
        self._record_quality_metrics(confidence_mask)

        return pseudo_labels, confidence_mask

    def _get_current_threshold(self):
        """动态调整阈值策略"""
        progress = self.current_iter / self.max_iters
        return self.base_threshold + (0.4 * (1 - np.cos(np.pi * progress)))

    def _create_confidence_mask(self, cos_sim, max_sim, threshold):
        """创建置信度掩码"""
        # 类间差异筛选
        sorted_sim = torch.sort(cos_sim, dim=1, descending=True)[0]
        class_gap = sorted_sim[:, 0] - sorted_sim[:, 1]
        gap_mask = class_gap > threshold

        # 绝对置信度筛选
        confidence_mask = max_sim > self.confidence_threshold

        # 组合掩码
        combined_mask = gap_mask & confidence_mask
        return combined_mask

    def _crf_postprocessing(self, features, pseudo_labels):
        """CRF后处理（简化的形态学操作替代）"""
        # 实际实现应使用CRF库，这里演示用开运算替代
        kernel = torch.ones(3, 3, device=self.device)
        refined_labels = []
        for batch in pseudo_labels:
            label = batch.float().unsqueeze(0).unsqueeze(0)
            label = F.max_pool2d(label, kernel_size=3, stride=1, padding=1)
            label = F.avg_pool2d(label, kernel_size=3, stride=1, padding=1)
            refined_labels.append(label.squeeze())
        return torch.stack(refined_labels)

    def _record_quality_metrics(self, confidence_mask):
        """记录伪标签质量指标"""
        coverage = torch.mean(confidence_mask.float()).item()
        self.pseudo_quality_history.append({
            'iteration': self.current_iter,
            'coverage': coverage,
            'threshold': self._get_current_threshold()
        })
        self.current_iter += 1

    def reset(self):
        """重置状态"""
        self.class_centers = torch.zeros_like(self.class_centers)
        self.updated_counts = torch.zeros_like(self.updated_counts)
        self.pseudo_quality_history = []
        self.current_iter = 0
