import copy
import zlib
import struct
import io
import clip
import torch
import numpy as np

def get_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity


def get_image_features(image, model, cpreprocess, device='cuda', need_preprocess=False):
    if need_preprocess:
        image = cpreprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

def freeze_param_mlp(model):
    for name, param in model.named_parameters():
        # if 'layers.0' in name or 'layers.1' in name or 'layers.3' in name:
        if 'layers.0' in name:
            param.requires_grad = False
def freeze_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
        # pass
    # Warning: Set it as True when using FedAVG, MOON, etc.

def freeze_param_norm(model):
    for name, param in model.named_parameters():
        # if 'layers.0' in name or 'layers.1' in name or 'layers.3' in name:
        if 'bn' in name or 'relu' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def get_text_features_list(texts, model, device='cuda', train=False):
    if train:
        text_inputs = torch.cat([clip.tokenize(c)
                                for c in texts]).to(device)
        text_features = model.encode_text(text_inputs)
    else:
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(c)
                                     for c in texts]).to(device)
            text_features = model.encode_text(text_inputs)

    return text_features


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def convert_models_to_fp32(model):
    for name, p in model.named_parameters():
        p.data = p.data.float()
        if p.grad == None:
            pass
        else:
            p.grad.data = p.grad.data.float()

import torch.nn as nn
import torch.nn.functional as F
class FocalLossWithSmoothing(nn.Module):
    def __init__(
            self,
            num_classes: int,
            gamma: int = 1,
            lb_smooth: float = 0.1,
            size_average: bool = True,
            ignore_index: int = None,
            alpha: float = None):
        """
        :param gamma:
        :param lb_smooth:
        :param ignore_index:
        :param size_average:
        :param alpha:
        """
        super(FocalLossWithSmoothing, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._size_average = size_average
        self._ignore_index = ignore_index
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._alpha = alpha

        if self._num_classes <= 1:
            raise ValueError('The number of classes must be 2 or higher')
        if self._gamma < 0:
            raise ValueError('Gamma must be 0 or higher')
        if self._alpha is not None:
            if self._alpha <= 0 or self._alpha >= 1:
                raise ValueError('Alpha must be 0 <= alpha <= 1')

    def forward(self, logits, label):
        """
        :param logits: (batch_size, class, height, width)
        :param label:
        :return:
        """
        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self._ignore_index is not None:
                ignore = label.eq(self._ignore_index)
                label[ignore] = 0
            lb_pos, lb_neg = 1. - self._lb_smooth, self._lb_smooth / (self._num_classes - 1)
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        """
        :param logits:
        :param label:
        :return:
        """
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level


class LinearDiscriminantLoss(nn.Module):
    def __init__(self, num_classes):
        super(LinearDiscriminantLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, features, labels):
        """
        features: 提取的特征，形状为 [batch_size, feature_dim]
        labels: 真实标签，形状为 [batch_size]
        """
        # 计算每个类别的特征均值
        unique_labels = torch.unique(labels)
        class_means = torch.zeros((self.num_classes, features.size(1))).to(features.device)
        for label in unique_labels:
            class_features = features[labels == label]
            class_means[label] = torch.mean(class_features, dim=0)

        # 计算总体均值
        overall_mean = torch.mean(features, dim=0)

        # 计算类间散射矩阵和类内散射矩阵
        between_class_scatter = torch.zeros((features.size(1), features.size(1))).to(features.device)
        within_class_scatter = torch.zeros((features.size(1), features.size(1))).to(features.device)
        for label in unique_labels:
            class_features = features[labels == label]
            class_mean = class_means[label]
            between_class_scatter += (class_mean - overall_mean).unsqueeze(1) @ (class_mean - overall_mean).unsqueeze(0)
            within_class_scatter += torch.sum((class_features - class_mean.unsqueeze(0)) @ ((class_features - class_mean.unsqueeze(0)).t()), dim=0)

        # 计算损失值
        loss = -torch.trace(between_class_scatter @ torch.inverse(within_class_scatter + 1e-6 * torch.eye(features.size(1)).to(features.device)))

        return loss


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

def get_one_hot(y_s: torch.tensor, num_classes: int):
    """
        args:
            y_s : torch.Tensor of shape [n_task, shot]
        returns
            y_s : torch.Tensor of shape [n_task, shot, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes] # [1, y_s.squeeze, num_classes]
    print('aa',list(y_s.size()) + [num_classes], y_s.size())
    one_hot = torch.zeros(one_hot_size, device=y_s.device, dtype=torch.float16) # [1, y_s.squeeze, num_classes]
    # print(one_hot)

    one_hot.scatter_(-1, y_s.unsqueeze(-1), 1)
    return one_hot


def compute_centroids_alpha(z_s: torch.tensor,
                      y_s: torch.tensor, num_classes):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    one_hot = get_one_hot(y_s, num_classes=num_classes) # [1, y_s.squeeze, num_classes]
    centroids = (one_hot*z_s/ one_hot.sum(-2, keepdim=True)).sum(1)  # [batch, K, d]
    centroids = torch.where(torch.isnan(centroids), torch.full_like(centroids, .0), centroids)
    # print(centroids)
    return centroids


def compute_centroids(z_s: torch.tensor,
                      y_s: torch.tensor, num_classes):
    """
    inputs:
        z_s : torch.Tensor of size [batch_size, s_shot, d]
        y_s : torch.Tensor of size [batch_size, s_shot]

    updates :
        centroids : torch.Tensor of size [n_task, num_class, d]
    """
    # print(z_s.shape)
    # print(y_s.shape)
    # print(y_s.unique().size(0))
    one_hot = get_one_hot(y_s, num_classes=num_classes).transpose(1, 2)
    # print(one_hot.shape)
    # centroids = one_hot.bmm(z_s) / one_hot.sum(-1, keepdim=True)  # [batch, K, d]
    centroids = one_hot.bmm(z_s)  # [batch, K, d], centroids shape: [1, num_classes, dim_features]
    # print(centroids.shape)
    return centroids




def distributed_sinkhorn(similarities, epsilon=0.7, num_iters=3):
    Q = torch.exp(similarities / epsilon).t()
    # print(Q)
    Q = Q / Q.sum(dim=0, keepdim=True)  # Normalize each column

    K, B = Q.shape  # Number of clusters and batch size , [Classes, Batch size]

    r = torch.ones(K).to(similarities.device) / K
    c = torch.ones(B).to(similarities.device) / B

    for _ in range(num_iters):
        u = Q.sum(dim=1)
        Q = Q * (r / (u + 1e-8)).unsqueeze(1)  # Row normalization

        v = Q.sum(dim=0)
        Q = Q * (c / (v + 1e-8)).unsqueeze(0)  # Column normalization

    Q = (Q / Q.sum(dim=0, keepdim=True)).t()

    # log_probs = F.log_softmax(logits, dim=-1)
    # loss_sinkhorn =   # this is a maximization problem
    # return -torch.mean(torch.sum(Q * log_probs, dim=1))
    return Q

from torchvision.transforms.functional import gaussian_blur

def add_poisson_noise(image_tensor):

    min_val = torch.min(image_tensor)
    max_val = torch.max(image_tensor)


    if min_val < 0:
        image_tensor = image_tensor - min_val  

    if max_val > 0:
        image_tensor = image_tensor / max_val * 255  

    noisy_image = torch.poisson(image_tensor)

    if max_val > 0:
        noisy_image = noisy_image / 255 * max_val  
    if min_val < 0:
        noisy_image = noisy_image + min_val  

    return noisy_image

def add_mixed_noise(image_tensor, gaussian_std=0.01, salt_pepper_prob=0.05, blur_kernel_size=5):
    # print(image_tensor)
    # image_tensor = image_tensor + torch.randn_like(image_tensor) * gaussian_std

    # salt = torch.rand_like(image_tensor) < (salt_pepper_prob / 2)
    # pepper = torch.rand_like(image_tensor) < (salt_pepper_prob / 2)
    # image_tensor[salt] = 1.0  
    # image_tensor[pepper] = 0.0 

    # image_tensor = add_poisson_noise(image_tensor)

    # image_tensor = torch.clamp(noisy_image, 0, 1)
    # print(noisy_image)
    return image_tensor


def compress_model(model):
    buffer = io.BytesIO()

    for name, param in model.fea_attn.state_dict().items():
        if 'dropout' in name:
            continue

        data = param.data.cpu().numpy()
        half_data = data.astype(np.float16)


        buffer.write(struct.pack('>I', len(name))) 
        buffer.write(name.encode('utf-8'))
        buffer.write(struct.pack('>I', half_data.ndim)) 
        for s in half_data.shape: 
            buffer.write(struct.pack('>I', s))
        buffer.write(half_data.tobytes())

    raw_data = buffer.getvalue()
    return zlib.compress(raw_data)


def decompress_model(compressed_data, model_template):
    raw_data = zlib.decompress(compressed_data)
    buffer = io.BytesIO(raw_data)

    model = copy.deepcopy(model_template) # model_template = ClipModelAt_quantize()
    model_state = model.fea_attn.state_dict()

    while buffer.tell() < len(raw_data):
        name_len = int(struct.unpack('>I', buffer.read(4))[ 0 ])
        name = buffer.read(name_len).decode('utf-8')
        ndim = int(struct.unpack('>I', buffer.read(4))[ 0 ])
        shape = tuple(int(struct.unpack('>I', buffer.read(4))[ 0 ]) for _ in range(ndim))

        data_size = int(np.prod(shape)) * 2  

        if name in model_state:
            byte_data = buffer.read(data_size)
            data = np.frombuffer(byte_data, dtype=np.float16).reshape(shape)
            model_state[ name ].copy_(torch.from_numpy(data.astype(np.float32)))
        else:
            buffer.read(data_size)  

    return model


def kl_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (p.clamp_min(1e-7).log() - q.clamp_min(1e-7).log())).sum(-1).mean()

# def crkd_loss(z_s: torch.Tensor, z_t: torch.Tensor, T: float = 2) -> torch.Tensor:
#     B, C = z_s.shape
#     loss = 0.0
#     for c in range(C):
#         t_logits = z_t[:, c] / T
#         s_logits = z_s[:, c] / T
#         t_prob = F.softmax(t_logits, dim=0)
#         s_prob = F.softmax(s_logits, dim=0)
#         loss += kl_div(s_prob, t_prob)
#     return loss / C

def crkd_loss(z_s: torch.Tensor, z_t: torch.Tensor, T: float = 2) -> torch.Tensor:
    B, C = z_s.shape
    loss = 0.0
    for c in range(C):
        t_logits = z_t[:, c] / T
        s_logits = z_s[:, c] / T
        t_prob = F.softmax(t_logits, dim=0)
        s_prob = F.softmax(s_logits, dim=0)
        loss += F.kl_div(s_prob.log(), t_prob, reduction='sum')
    return loss / C




def population_norm_regularizer(mu, gamma_sq, x):
    # x 可能是 [B, C] 或 [B, C, H, W]
    if x.dim() == 2:                       # MLP 情形
        mean_x = x.mean(dim=0)           # [C]
        var_x  = ((x - mu.view(1, -1)) ** 2).mean(dim=0)
    elif x.dim() == 4:                     # CNN 情形
        mean_x = x.mean(dim=(0, 2, 3))   # [C]
        var_x  = ((x - mu.view(1, -1, 1, 1)) ** 2).mean(dim=(0, 2, 3))
    else:
        raise ValueError("Unsupported tensor shape")

    mean_loss = F.mse_loss(mu, mean_x)
    var_loss  = F.mse_loss(gamma_sq, var_x)
    return mean_loss + var_loss
