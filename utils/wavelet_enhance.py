import pywt
import torch
import torch.nn as nn
import numpy as np

class WaveletEnhancement(nn.Module):
    def __init__(self, wavelet='db4', levels=2, enhance_factor=1.5, denoise_threshold=0.2):
        super(WaveletEnhancement, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.enhance_factor = enhance_factor  # 小波系数增强因子
        self.denoise_threshold = denoise_threshold  # 去噪阈值

    def forward(self, x):
        """
        对输入图像进行多尺度小波增强，并添加特征扰动
        :param x: 输入图像张量，形状为 (B, C, H, W)
        :return: 增强后的图像张量，形状为 (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # 转换为 (B, H, W, C) 的 numpy 数组

        enhanced_x = []
        for i in range(B):
            img = x[i]  # (H, W, C)
            enhanced_channels = []
            for c in range(C):
                # 对每个通道进行小波分解
                coeffs = pywt.wavedec2(img[:, :, c], wavelet=self.wavelet, level=self.levels)
                enhanced_coeffs = []
                for level in range(self.levels + 1):
                    if level == 0:
                        # 低频系数不进行增强
                        enhanced_coeffs.append(coeffs[level])
                    else:
                        # 对高频系数进行增强和去噪
                        cH, cV, cD = coeffs[level]
                        # 放大高频系数来增强细节信息
                        enhanced_cH = cH * self.enhance_factor
                        enhanced_cV = cV * self.enhance_factor
                        enhanced_cD = cD * self.enhance_factor

                        # 进行去噪处理
                        enhanced_cH = pywt.threshold(enhanced_cH, self.denoise_threshold * np.max(np.abs(enhanced_cH)), mode='soft')
                        enhanced_cV = pywt.threshold(enhanced_cV, self.denoise_threshold * np.max(np.abs(enhanced_cV)), mode='soft')
                        enhanced_cD = pywt.threshold(enhanced_cD, self.denoise_threshold * np.max(np.abs(enhanced_cD)), mode='soft')

                        enhanced_coeffs.append((enhanced_cH, enhanced_cV, enhanced_cD))
                # 小波重构
                enhanced_img_channel = pywt.waverec2(enhanced_coeffs, wavelet=self.wavelet)
                enhanced_channels.append(enhanced_img_channel)
            # 合并增强后的通道
            enhanced_img = np.stack(enhanced_channels, axis=-1)
            enhanced_x.append(enhanced_img)

        # 将增强后的图像转换回张量
        enhanced_x = np.stack(enhanced_x, axis=0)  # (B, H, W, C)
        enhanced_x = torch.from_numpy(enhanced_x).permute(0, 3, 1, 2).contiguous().float().cuda()

        return enhanced_x


