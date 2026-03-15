import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def cross_entropy_2d(pred,label,cfg):

    '''
    Args:
        predict:(n, c, h, w)
        target: (n, h, w)
    '''
    assert not label.requires_grad
    assert pred.dim()   == 4
    assert label.dim()  == 3
    assert pred.size(0) == label.size(0), f'{pred.size(0)}vs{label.size(0)}'
    assert pred.size(2) == label.size(1), f'{pred.size(2)}vs{label.size(2)}'
    assert pred.size(3) == label.size(2), f'{pred.size(3)}vs{label.size(3)}'

    n,c,h,w = pred.size()
    label   = label.view(-1)


    class_count = torch.bincount(label).float()
    try:

        assert class_count.size(0) == 5
        new_class_count = class_count
    except:
        new_class_count = torch.zeros(5).cuda().float()
        new_class_count[:class_count.size(0)] = class_count

    # organ_all_count = new_class_count[1]+new_class_count[2]+new_class_count[3]+new_class_count[4]
    # weight      = (1 - (new_class_count+1)/organ_all_count)
    # print(new_class_count)
    # weight[0] = (1 - (new_class_count+1)/label.size(0))[0]
    weight = (1 - (new_class_count + 1) / label.size(0))
    # print(weight)
    pred    = pred.transpose(1,2).transpose(2,3).contiguous() #n*c*h*w->n*h*c*w->n*h*w*c
    pred    = pred.view(-1,c)
    loss    = F.cross_entropy(input=pred,target=label,weight=weight)
    return loss

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

def dice_loss(pred, target):
    """
    input is a torch variable of size [N,C,H,W]
    target: [N,H,W]
    """
    target = target.long()
    n,c,h,w        = pred.size()
    pred           = pred.cuda()
    target         = target.cuda()
    target_onehot  = torch.zeros([n,c,h,w]).cuda()

    target         = torch.unsqueeze(target,dim=1) # n*1*h*w
    target_onehot.scatter_(1,target,1)

    assert pred.size() == target_onehot.size(), "Input sizes must be equal."
    assert pred.dim()  == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target_onehot.cpu().data.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    eps = 1e-5
    probs = F.softmax(pred,dim=1)
    num   = probs * target_onehot  # b,c,h,w--p*g
    num   = torch.sum(num, dim=3)  # b,c,h
    num   = torch.sum(num, dim=2)  # b,c,

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)  # b,c,

    den2 = target_onehot * target_onehot  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2.0 * (num / (den1 + den2+eps))  # b,c

    dice_total =  torch.sum(dice) / dice.size(0)  # divide by batch_sz
    return 1 - 1.0 * dice_total/5.0

def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

class MPCL(nn.Module):
    def __init__(self, num_class=5,temperature=0.07,m=0.5,
                 base_temperature=0.07,easy_margin=False):
        super(MPCL, self).__init__()
        self.num_class        = num_class
        self.temperature      = temperature
        self.base_temperature = base_temperature
        self.m                = m
        self.cos_m            = math.cos(m)
        self.sin_m            = math.sin(m)
        self.th               = math.cos(math.pi - m)
        self.mm               = math.sin(math.pi - m) * m
        self.easy_margin      = easy_margin

    def forward(self, features, labels,class_center_feas,
                pixel_sel_loc=None, mask=None):
        """

         features: [batch_size*fea_h*fea_w] * 1 *c  normalized
         labels:   batch_size*fea_h*fea_w
         class_center_feas:  n_fea*n_class  normalized

        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # build mask
        num_samples = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(num_samples, dtype=torch.float32).cuda()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1).long()  # n_sample*1
            class_center_labels = torch.range(0,self.num_class-1).long().cuda()
            # print(class_center_labels)
            class_center_labels = class_center_labels.contiguous().view(-1,1) # n_class*1
            if labels.shape[0] != num_samples:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels,torch.transpose(class_center_labels,0,1)).float().cuda() # broadcast n_sample*n_class
        else:
            mask = mask.float().cuda()
        # n_sample = batch_size * fea_h * fea_w
        # mask n_sample*n_class  the mask_ij represents whether the i-th sample has the same label with j-th class or not.
        # in our experiment, the n_view = 1, so the contrast_count = 1
        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [n*h*w]*fea_s

        anchor_feature = contrast_feature
        anchor_count   = contrast_count
        # compute logits
        cosine = torch.matmul(anchor_feature, class_center_feas) # [n*h*w] * n_class
        logits = torch.div(cosine,self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits        = logits - logits_max.detach()

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0.0001, 1.0))
        phi  = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # print(phi)
        phi_logits = torch.div(phi,self.temperature)

        phi_logits_max, _ = torch.max(phi_logits, dim=1, keepdim=True)
        phi_logits = phi_logits - phi_logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        tag_1             = (1-mask)
        tag_2             = mask
        exp_logits        = torch.exp(logits*tag_1 + phi_logits * tag_2)
        phi_logits        = (logits*tag_1) + (phi_logits*tag_2)
        log_prob          = phi_logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-4)


        if pixel_sel_loc is not None:

            pixel_sel_loc     = pixel_sel_loc.view(-1)

            mean_log_prob_pos =  (mask * log_prob).sum(1)
            mean_log_prob_pos = pixel_sel_loc * mean_log_prob_pos
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = torch.div(loss.sum(),pixel_sel_loc.sum()+1e-4)
        else:

            mean_log_prob_pos = (mask * log_prob).sum(1)
            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, num_samples).mean()

        return loss

# MMD损失
class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(3)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(3)))
        L2_distance = ((total0 - total1) ** 2).sum(dim=[2, 3, 4])  # 对所有特征维度求和
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class EnhancedMMDLoss(nn.Module):
    """
    改进的MMD损失函数，具有以下增强特性：
    1. 自适应多核选择：根据特征分布动态调整核函数
    2. 类别感知MMD：利用类别信息进行细粒度对齐
    3. 动态权重调整：根据训练进度调整损失权重
    4. 特征归一化：增强数值稳定性

    参数:
        kernel_type: 核函数类型('rbf', 'linear'等)
        num_classes: 类别数量
        base_temperature: 基础温度参数
        dynamic_weight: 是否使用动态权重
        class_aware: 是否使用类别感知MMD
    """

    def __init__(self, kernel_type='rbf', num_classes=5, base_temperature=1.0,
                 dynamic_weight=True, class_aware=True):
        super(EnhancedMMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.num_classes = num_classes
        self.base_temperature = base_temperature
        self.dynamic_weight = dynamic_weight
        self.class_aware = class_aware

        # 核函数参数
        self.kernel_mul = 2.0
        self.kernel_num = 7
        self.fix_sigma = None

    def compute_kernels(self, x, y):
        """
        计算多核MMD的核矩阵
        """
        n_x = x.size(0)
        n_y = y.size(0)

        # 计算样本间距离
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())

        # 计算距离矩阵
        xx_diag = torch.diag(xx).unsqueeze(0).expand_as(xx)
        yy_diag = torch.diag(yy).unsqueeze(0).expand_as(yy)

        xx_dist = xx_diag + xx_diag.t() - 2 * xx
        yy_dist = yy_diag + yy_diag.t() - 2 * yy
        xy_dist = xx_diag.t() + yy_diag - 2 * xy

        # 使用中位数启发式设置带宽
        if self.fix_sigma is None:
            all_dist = torch.cat([xx_dist.view(-1), yy_dist.view(-1), xy_dist.view(-1)])
            median_dist = torch.median(all_dist[all_dist > 0])  # 忽略零距离
            base_sigma = median_dist * 2.0  # 更稳定的带宽估计
            base_sigma = base_sigma if base_sigma != 0 else 1.0
        else:
            base_sigma = self.fix_sigma

        # 创建多尺度核
        sigma_list = [base_sigma * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernels = []

        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma)
            k_xx = torch.exp(-gamma * xx_dist)
            k_yy = torch.exp(-gamma * yy_dist)
            k_xy = torch.exp(-gamma * xy_dist)
            kernels.append((k_xx, k_yy, k_xy))

        return kernels

    def forward(self, source, target, source_labels=None, target_pseudo_labels=None,
                current_iter=0, max_iter=10000):
        """
        计算改进的MMD损失

        参数:
            source: 源域特征 [B, C, H, W] 或 [B, D]
            target: 目标域特征 [B, C, H, W] 或 [B, D]
            source_labels: 源域标签 [B, H, W]
            target_pseudo_labels: 目标域伪标签 [B, H, W]
            current_iter: 当前迭代步数
            max_iter: 最大迭代步数
        """
        # 特征归一化
        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

        # 展平空间维度
        if source.dim() > 2:
            B, C, H, W = source.shape
            source = source.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            target = target.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

            if source_labels is not None:
                source_labels = source_labels.view(-1)  # [B*H*W]
            if target_pseudo_labels is not None:
                target_pseudo_labels = target_pseudo_labels.view(-1)  # [B*H*W]

        if self.class_aware and (source_labels is not None) and (target_pseudo_labels is not None):
            # 类别感知MMD - 按类别分别计算
            class_mmd = 0
            valid_classes = 0

            for cls_idx in range(self.num_classes):
                # 获取当前类别的源域和目标域样本
                src_cls_mask = (source_labels == cls_idx)
                tgt_cls_mask = (target_pseudo_labels == cls_idx)

                if torch.any(src_cls_mask) and torch.any(tgt_cls_mask):
                    src_cls_feat = source[src_cls_mask]
                    tgt_cls_feat = target[tgt_cls_mask]

                    # 为当前类别计算MMD
                    kernels = self.compute_kernels(src_cls_feat, tgt_cls_feat)
                    cls_mmd = 0

                    for k_xx, k_yy, k_xy in kernels:
                        cls_mmd += k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

                    # 根据类别样本数量加权
                    cls_weight = min(src_cls_mask.sum(), tgt_cls_mask.sum()).float()
                    class_mmd += cls_mmd * cls_weight
                    valid_classes += 1

            if valid_classes > 0:
                mmd_val = class_mmd / (valid_classes * self.kernel_num)
            else:
                mmd_val = torch.tensor(0.0, device=source.device)
        else:
            # 标准MMD计算
            kernels = self.compute_kernels(source, target)
            mmd_val = 0

            for k_xx, k_yy, k_xy in kernels:
                mmd_val += k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

            mmd_val /= self.kernel_num

        # 动态权重调整
        if self.dynamic_weight:
            # 训练后期减小MMD权重，让任务损失主导
            weight = 1.0 - (current_iter / max_iter) * 0.5
            mmd_val *= weight

        return mmd_val
def loss_calc_pro(pred,label,cfg):

    '''
    This function returns cross entropy loss for semantic segmentation
    '''
    # pred shape is batch * c * h * w
    # label shape is b*h*w

    label = label.long().cuda()
    return cross_entropy_2d_pro(pred, label,cfg)

def cross_entropy_2d_pro(pred, label, cfg):
    '''
    Args:
        predict:(n, c, h, w)
        target: (n, h, w)
    '''
    assert not label.requires_grad
    assert pred.dim()   == 4
    assert label.dim()  == 3
    assert pred.size(0) == label.size(0), f'{pred.size(0)}vs{label.size(0)}'
    assert pred.size(2) == label.size(1), f'{pred.size(2)}vs{label.size(2)}'
    assert pred.size(3) == label.size(2), f'{pred.size(3)}vs{label.size(3)}'


    n,c,h,w = pred.size()

    label   = label.view(-1)
    #adjusted_label = torch.where(label == 4, 3, label)
    #class_count = torch.bincount(adjusted_label).float()
    class_count = torch.bincount(label).float()

    try:

        assert class_count.size(0) == c
        #assert class_count.size(0) == 2
        new_class_count = class_count
    except:
        new_class_count = torch.zeros(c).cuda().float()
        #new_class_count = torch.zeros(2).cuda().float()
        new_class_count[:class_count.size(0)] = class_count

    # organ_all_count = new_class_count[1]+new_class_count[2]+new_class_count[3]+new_class_count[4]
    # weight      = (1 - (new_class_count+1)/organ_all_count)
    # print(new_class_count)
    # weight[0] = (1 - (new_class_count+1)/label.size(0))[0]
    weight = (1 - (new_class_count + 1) / label.size(0))
    #weight = torch.tensor([0.1, 1.0, 1.0, 1.0]).cuda()  # 假设背景是第0类

    # print(weight)
    pred    = pred.transpose(1,2).transpose(2,3).contiguous() #n*c*h*w->n*h*c*w->n*h*w*c
    pred    = pred.view(-1,c)

    loss    = F.cross_entropy(input=pred,target=label,weight=weight)
    return loss
def dice_loss_pro(pred, target):
    """
    input is a torch variable of size [N,C,H,W]
    target: [N,H,W]
    """
    target = target.long()
    n,c,h,w        = pred.size()
    pred           = pred.cuda()
    target         = target.cuda()
    target_onehot  = torch.zeros([n,c,h,w]).cuda()

    target         = torch.unsqueeze(target,dim=1) # n*1*h*w
    target_onehot.scatter_(1,target,1)

    assert pred.size() == target_onehot.size(), "Input sizes must be equal."
    assert pred.dim()  == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target_onehot.cpu().data.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    eps = 1e-5
    probs = F.softmax(pred,dim=1)
    num   = probs * target_onehot  # b,c,h,w--p*g
    num   = torch.sum(num, dim=3)  # b,c,h
    num   = torch.sum(num, dim=2)  # b,c,

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)  # b,c,

    den2 = target_onehot * target_onehot  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2.0 * (num / (den1 + den2+eps))  # b,c

    dice_total =  torch.sum(dice) / dice.size(0)  # divide by batch_sz
    return 1 - 1.0 * dice_total/2.0


class SemanticConsistencyLoss(nn.Module):
    def __init__(self, num_classes):
        super(SemanticConsistencyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, src_fea, trg_fea, src_labels, trg_pseudo_labels):
        class_center_src = self.compute_class_centers(src_fea, src_labels)
        class_center_trg = self.compute_class_centers(trg_fea, trg_pseudo_labels)
        similarity_loss = torch.norm(class_center_src - class_center_trg, p=2)
        return similarity_loss

    def compute_class_centers(self, features, labels):
        class_centers = torch.zeros(self.num_classes, features.size(1)).cuda()
        for c in range(self.num_classes):
            class_mask = (labels == c).float()
            class_features = features * class_mask.unsqueeze(1)
            class_center = class_features.sum(dim=(2, 3)) / (class_mask.sum(dim=(1, 2)) + 1e-8)
            class_centers[c] = class_center.mean(dim=0)
        return class_centers

def consistency_loss_kl(pred_main, pred_main_aug):
    # 使用 KL 散度计算一致性损失
    kl_distance = nn.KLDivLoss(reduction='mean')
    log_softmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    consistency = kl_distance(log_softmax(pred_main), softmax(pred_main_aug))
    return consistency

def consistency_loss_kl_symmetric(pred1, pred2, epsilon=1e-8):
    pred1_prob = F.softmax(pred1, dim=1) + epsilon
    pred2_prob = F.softmax(pred2, dim=1) + epsilon
    kl_div1 = F.kl_div(torch.log(pred1_prob), pred2_prob, reduction='mean')
    kl_div2 = F.kl_div(torch.log(pred2_prob), pred1_prob, reduction='mean')
    return (kl_div1 + kl_div2) / 2.0


# 频域特征约束损失函数（作用于特征图）
class FeatureFrequencyConstraintLoss(nn.Module):
    def __init__(self):
        super(FeatureFrequencyConstraintLoss, self).__init__()

    def forward(self, source_fea, target_fea):
        source_fft = torch.fft.fftn(source_fea, dim=(-2, -1))
        target_fft = torch.fft.fftn(target_fea, dim=(-2, -1))
        source_fft_abs = torch.abs(source_fft)
        target_fft_abs = torch.abs(target_fft)
        source_fft_phase = torch.angle(source_fft)
        target_fft_phase = torch.angle(target_fft)
        amp_diff = torch.mean(torch.abs(source_fft_abs - target_fft_abs) / (source_fft_abs + 1e-6))
        phase_diff = torch.mean(torch.abs(source_fft_phase - target_fft_phase))
        freq_diff = amp_diff + phase_diff
        return freq_diff


