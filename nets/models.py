import torch.nn as nn
import clip
import open_clip
from torch.nn import LeakyReLU
import math
from utils.clip_util import freeze_param, get_image_features
import torch
import torch.nn.functional as F
import loraclip
import timm


class ClipModelat_openclip(nn.Module):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px',
        'ViT-H/14',
        'EVA02-B/16',
        'convnext_base',
        'convnext_large_d',
        'MobileCLIP-S1',
        'ViT-g/14',
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, attention=True, freezepy=True):
        super(ClipModelat_openclip, self).__init__()
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained='laion2b_s26b_b102k_augreg')
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.attention = attention
        self.freezepy = freezepy
        self.device = device

    def initdgatal(self, dataloader):

        for batch in dataloader:
            with torch.no_grad():
                image, _, label = batch
                image = image.to(self.device)
                label = label.to(self.device)
                image_features = self.model.encode_image(image)
                break
        if self.freezepy:
            freeze_param(self.model)

        if self.attention:
            # pass
            self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
                                          nn.InstanceNorm1d(image_features.shape[1]),
                                          nn.ReLU6(), nn.Linear(image_features.shape[1], image_features.shape[1]),
                                          nn.Softmax(dim=1)).to(self.device)
            # For FedClip
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            # nn.Tanh(), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)


    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        # print(labels)
        self.labels = labels

class ClipModelat(nn.Module):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, attention=True, freezepy=True):
        super(ClipModelat, self).__init__()
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(
            model_name, device=device)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.attention = attention
        self.freezepy = freezepy
        self.device = device

    def initdgatal(self, dataloader):

        for batch in dataloader:
            with torch.no_grad():
                image, _, label = batch
                image = image.to(self.device)
                label = label.to(self.device)
                # print(label)
                image_features = self.model.encode_image(image)
                break
        if self.freezepy:
            freeze_param(self.model)

        if self.attention:
            self.fea_attn = nn.Sequential(MaskedMLP(image_features.shape[1], image_features.shape[1]), nn.BatchNorm1d(image_features.shape[1]),
                           nn.ReLU(), MaskedMLP(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)
            # self.fea_attn = Global_attention_block(image_features)
            # For FedClip
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            # nn.Tanh(), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)
            # For FACMIC
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            #                               nn.BatchNorm1d(image_features.shape[1]),
            #                               nn.LeakyReLU(), nn.Linear(image_features.shape[1], image_features.shape[1]),
            #                               nn.Softmax(dim=1)).to(self.device)
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            #                               , nn.LeakyReLU(
            #     ), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.BatchNorm1d(image_features.shape[1]),
            #                               nn.LeakyReLU(
            #     ), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)


    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        # print(labels)
        self.labels = labels


class ClipModel_mamba(nn.Module):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, attention=True, freezepy=True):
        super(ClipModel_mamba, self).__init__()
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(
            model_name, device=device)
        self.visual = timm.create_model('mambaout_base_plus_rw.sw_e150_in12k_ft_in1k', pretrained=True,num_classes=0)
        self.visual_projector = torch.nn.Linear(3072, 512)
        self.visual.eval()
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.attention = attention
        self.freezepy = freezepy
        self.device = device

    def initdgatal(self, dataloader):

        for batch in dataloader:
            with torch.no_grad():
                image, _, label = batch
                image = image.to(self.device)
                label = label.to(self.device)
                # print(label)
                image_features = self.model.encode_image(image)
                break
        if self.freezepy:
            freeze_param(self.model)

        if self.attention:
            self.fea_attn = nn.Sequential(MaskedMLP(image_features.shape[1], image_features.shape[1]), PopulationNorm(image_features.shape[1]),
                           nn.ReLU(), MaskedMLP(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)
            # self.fea_attn = Global_attention_block(image_features)
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            #                               , nn.LeakyReLU(
            #     ), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.BatchNorm1d(image_features.shape[1]),
            #                               nn.LeakyReLU(
            #     ), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)


    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        # print(labels)
        self.labels = labels


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        # self.linear = nn.Linear(input_size, hidden_size)
        # self.bn = nn.BatchNorm1d(hidden_size)
        # self.relu = AReLU()  ## AReLU()
        # self.relu2 = AReLU()
        # self.relu3 = AReLU()
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        # self.fc4 = nn.Linear(hidden_size, num_classes)
        self.linear = MaskedMLP(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()  ## AReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.linear2 = MaskedMLP(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.linear3 = MaskedMLP(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = MaskedMLP(hidden_size, num_classes) # before it was fc4.

    def forward(self, x):
        x = self.linear(x)
        # x = self.attn(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = self.attn(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        # x = self.attn(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

        # return self.layers(x)


class MLP_old(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP_old, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), # 0
            nn.BatchNorm1d(hidden_size),  # 1
            nn.LeakyReLU(), # 2
            nn.Linear(hidden_size, hidden_size),  # 3
            nn.BatchNorm1d(hidden_size), # 4
            nn.LeakyReLU(), # 5
            nn.Linear(hidden_size, hidden_size), # 6
            nn.BatchNorm1d(hidden_size), # 7
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_classes), # 9
            # nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class TempPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # 输出范围 (0,1)
        )

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        # z_s, z_t: [B, C] => mean over batch
        z_s = z_s.mean(0)      # [C]
        z_t = z_t.mean(0)      # [C]
        x = torch.cat([z_s, z_t])
        return self.net(x).squeeze()  # scalar in (0,1)

class ClipModelatLoRA(nn.Module):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, rank=4, freezepy=True):
        super(ClipModelatLoRA, self).__init__()
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = loraclip.load(
            model_name, device=device, r=rank, lora_mode="vision+text")
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.attention = True
        self.freezepy = freezepy
        self.device = device

    def initdgatal(self, dataloader):

        for batch in dataloader:
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = get_image_features(
                image, self.model, self.preprocess)
            break
        if self.freezepy:
            freeze_param(self.model)

        if self.attention:
            # pass
            self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]), nn.BatchNorm1d(image_features.shape[1]),
                           nn.LeakyReLU(),  nn.Linear(image_features.shape[1], image_features.shape[1]),  nn.Softmax(dim=1)).to(self.device)

    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        self.labels = labels


class ClientClassifier(nn.Module):
    def __init__(self, input_size, num_clients):
        super(ClientClassifier, self).__init__()
        # self.fc1 = nn.Linear(input_size, input_size)
        # self.attn = MultiHeadLatentAttention(input_size, num_heads=8, latent_dim=8, dropout=0.7)
        # self.norm1 = nn.LayerNorm(input_size)
        # self.fc2 = nn.Linear(input_size, input_size // 2)
        # self.norm2 = nn.LayerNorm(input_size // 2)
        # self.fc3 = nn.Linear(input_size // 2, num_clients)
        # self.relu = nn.ReLU6()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size), # 512, 512
            nn.InstanceNorm1d(input_size),  # 1
            nn.ReLU6(inplace=False),
            nn.Linear(input_size, input_size // 2),  # 512, 512
            nn.InstanceNorm1d(input_size // 2),  # 1
            nn.ReLU6(inplace=False),
            nn.Linear(input_size // 2, num_clients), # 9
        )

    def forward(self, x):
        # x = self.relu(self.fc1(x))
        # x = self.attn(x)
        # x = self.norm1(x)
        # x = self.relu(self.fc2(x))
        # x = self.norm2(x)

        # return self.fc3(x)
        return self.layers(x) # [B, N]


class ClipModelat_quantize(nn.Module):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, rank=4, freezepy=True):
        super(ClipModelat_quantize, self).__init__()
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(
            model_name, device=device)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.attention = True
        self.freezepy = freezepy
        self.device = device
        self.fea_attn = nn.Sequential(nn.Linear(512, 512), nn.InstanceNorm1d(512),
                           nn.ReLU6(),  nn.Linear(512, 512),  nn.Softmax(dim=1)).to(self.device)

    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        # print(labels)
        self.labels = labels

class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha


class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.01).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # print(input)
        zero_index = torch.abs(input) > 1
        # print(zero_index)
        grad_input = grad_output.clone()
        return grad_input*zero_index

class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        # self.bias = None
        self.threshold = nn.Parameter(torch.Tensor(out_size), requires_grad=True) #This was it, we have to share it right ? Linear layer only threshold needs to share
        self.step = BinaryStep.apply #it becomes forward embedded with specified custom backward
        self.mask = torch.ones(out_size, in_size, requires_grad=True)
        self.ratio = 1
        self.reset_parameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zero_count = 0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            #std = self.weight.std()
            self.threshold.data.fill_(0.) # first round all be 0

    def mask_generation(self, weight, thresholds):
        abs_weight = torch.abs(self.weight)
        abs_weight_mean = torch.mean(abs_weight, 1)
        abs_weight_mean = abs_weight_mean.view(abs_weight.shape[0], -1)
        threshold = thresholds.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight_mean - threshold
        # print(threshold)
        abs_weight = abs_weight.repeat(1, weight.shape[1])
        mask = self.step(abs_weight)
        self.mask = mask.to(self.device)
        self.ratio = torch.sum(self.mask) / self.mask.numel()

    def forward(self, input):
        # print('current self.ratio:', self.ratio)
        self.threshold.clip(-5, 5)
        if self.ratio <= 0.01:
            with torch.no_grad():
                #std = self.weight.std()
                self.threshold.data.fill_(0.)
                self.zero_count +=1
            self.mask_generation(self.weight, self.threshold)

        self.mask_generation(self.weight, self.threshold) #generate binary masks, self.threshold is learnable
        masked_weight = self.weight * self.mask
        output = torch.nn.functional.linear(input, masked_weight, self.bias)
        return output

def print_layer_keep_ratio(args, model):
    total = 0.
    keep = 0.
    for layer in model.modules():
        if isinstance(layer, MaskedMLP):
            abs_weight = torch.abs(layer.weight)
            threshold = layer.threshold.view(abs_weight.shape[0], -1)
            # print(abs_weight-threshold)
            abs_weight = abs_weight-threshold
            mask = layer.step(abs_weight)
            # print(mask)
            ratio = torch.sum(mask) / mask.numel() #torch.tensor.numel() returns the number of elements
            total += mask.numel()
            keep += torch.sum(mask)
            # logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            print("client_{}, mlp layer: {}, keep ratio {:.4f}".format(args.index, layer, ratio))
