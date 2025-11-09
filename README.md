Codes for our approach and other baselines.

**Below we provide additional results (e.g., calibration) on BraTS, ICH and Prostate datasets. If it is not viewable, please go to images folder (it is a problem of Anonymous Github).**

**BraTS dataset (Columns 1-5 represent clients and the global site.)**
<p align="center">
<img src="./images/BraTS/Ours/C0_ours.png" alt="intro" width="17%"/> <img src="./images/BraTS/Ours/C1_ours.png" alt="intro" width="17%"/> <img src="./images/BraTS/Ours/C2_ours.png" alt="intro" width="17%"/> <img src="./images/BraTS/Ours/C3_ours.png" alt="intro" width="17%"/>
 <img src="./images/BraTS/Ours/C_glo_ours.png" alt="intro" width="17%"/>
</p>

**Prostate dataset (Columns 1-4 represent clients and the global site.)**
<p align="center">
<img src="./images/Prostate/Ours/Client_local_0_reliability_diagram.png" alt="intro" width="23%"/> <img src="./images/Prostate/Ours/Client_local_1_reliability_diagram.png" alt="intro" width="23%"/><img src="./images/Prostate/Ours/Client_local_2_reliability_diagram.png" alt="intro" width="23%"/><img src="./images/Prostate/Ours/Client_local_3_reliability_diagram.png" alt="intro" width="23%"/>
</p>

**Brain_ICH dataset (5clients, First row represents client1 to client5.))**
<p align="center">
<img src="./images/brain_ICH/5clients/Ours/Client_local_0_reliability_diagram.png" alt="intro" width="17%"/> <img src="./images/brain_ICH/5clients/Ours/Client_local_1_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/5clients/Ours/Client_local_2_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/5clients/Ours/Client_local_3_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/5clients/Ours/Client_local_4_reliability_diagram.png" alt="intro" width="17%"/>
</p>

**Brain_ICH dataset (10clients, First row represents client1 to client5, while second row represents client6 to client 10.))**
<p align="center">
<img src="./images/brain_ICH/10clients/Ours/Client_local_0_reliability_diagram.png" alt="intro" width="17%"/> <img src="./images/brain_ICH/10clients/Ours/Client_local_1_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/10clients/Ours/Client_local_2_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/10clients/Ours/Client_local_3_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/10clients/Ours/Client_local_4_reliability_diagram.png" alt="intro" width="17%"/>
</p>
<p align="center">
<img src="./images/brain_ICH/10clients/Ours/Client_local_5_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/10clients/Ours/Client_local_6_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/10clients/Ours/Client_local_7_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/10clients/Ours/Client_local_8_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/10clients/Ours/Client_local_9_reliability_diagram.png" alt="intro" width="17%"/>
</p>

**Brain_ICH dataset (15clients, First row represents client1 to client5, second row represents client6 to client 10, third row means client11 to client 15.)**
<p align="center">
<img src="./images/brain_ICH/15clients/Ours/Client_local_0_reliability_diagram.png" alt="intro" width="17%"/> <img src="./images/brain_ICH/15clients/Ours/Client_local_1_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_2_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_3_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_4_reliability_diagram.png" alt="intro" width="17%"/>
</p>
<p align="center">
<img src="./images/brain_ICH/15clients/Ours/Client_local_5_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_6_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_7_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_8_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_9_reliability_diagram.png" alt="intro" width="17%"/>
</p>
<p align="center">
<img src="./images/brain_ICH/15clients/Ours/Client_local_10_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_11_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_12_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_13_reliability_diagram.png" alt="intro" width="17%"/><img src="./images/brain_ICH/15clients/Ours/Client_local_14_reliability_diagram.png" alt="intro" width="17%"/>
</p>

**P-test on BraTS, Prostate and brain_ICH datasets.**
<p align="center">
<img src="./images/PValue/Ours/P_Value_BraTS.png" alt="intro" width="32%"/> <img src="./images/PValue/Ours/P_Value_Prostate.png" alt="intro" width="32%"/>
</p>

**Test F1 and accuracy on ISIC2019 and BraTS datasets.**
<p align="center">
<img src="./images/F1_ISIC2019.png" alt="intro" width="49%"/><img src="./images/ACC_BraTS.png" alt="intro" width="49%"/>
</p>


## Requirements

We suggest you to use the following packages:

clip==1.0

loraclip==0.1.0

numpy==1.22.0

opencv-python==4.9.0.80

openpyxl==3.1.2

Pillow==9.3.0

scikit-image==0.21.0

scikit-learn==1.1.3

scipy==1.10.0

tqdm==4.66.1

torch==1.13.1+cu117

torchvision=0.14.1+cu117


## Tutorials

Here we provide a detailed instruction about how to run the codes. Here we use Our method as an example.

1) Running.
```sh
 python Ours.py
```
This can be done in the terminal. Since our code requires parsers, we set all default values, so there is no need to add values in python Ours.py.

2) Important Hyperparameters.

```sh
parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BraTS')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--root_dir', type=str, default='./data')
    parser.add_argument('--iters', type=int, default=50,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--net', type=str, default='ViT-B/32',
                        help='[RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14@336px]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[4])
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--step', type=float, default=0)
    parser.add_argument('--aggmode', type=str, default='att')
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--method', type=str, default='clip')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--lr_mlp', type=float, default=1e-3)
```

```sh
--dataset: Data we will use.  
--batch: Batch size.  
--root_dir: Root directory of the dataset.  
--iters: federated epochs.  
--wk_iters: local epoch.  
--net: CLIP architecture to use.
--seed: Random seed for reproducibility.   
--test_envs: Global site index.  
--beta1: Beta1 parameter for the AdamW optimizer.  
--beta2: Beta2 parameter for the AdamW optimizer.  
--eps: Epsilon value for numerical stability in the AdamW optimizer.  
--step: to store the current step.  
--aggmode: Aggregation mode for federated learning. (att means PEFT approaches such as FedCLIP, while AVG represent traditional methods such as FedAVG) 
--weight_decay: Weight decay coefficient for the AdamW optimizer.  
--method: Method to use for training (e.g., fedclip, lora, xx).
--lr: Learning rate of FAM.  
--lr_mlp: Learning rate of the local MLP.
```

3) Datasets.

The structures of our dataset are as follows:

```sh
./data/ISIC2019/
    C_1/
      nevus/
        frame_0001.jpg
            ...
      melanoma/
      squamous cell carcinoma/
      .../
    C_2/
    C_3/
    C_4/
```

C_1, C_2, ... represent each client, respectively. Nevus, melanoma and squamous cell carcinoma mean the class name.

4) Data preparations (train vs. val vs. test)

You can modify the data partition and processing techniques in prepare_data_dg_clip.py. You can define the percentage for training, val and test via:
```sh
l1, l2, l3 = int(l*0.6), int(l*0.2), int(l*0.2), Line 263.
```

5) Definitions of FAM and MLP.

./nets/models.py is the model backbone file. Our FAM is defined in Line 52-53 as follows.

```sh
if self.attention:
  self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]), nn.InstanceNorm1d(image_features.shape[1]),nn.ReLU6(),
  nn.Linear(image_features.shape[1], image_features.shape[1]),  nn.Softmax(dim=1)).to(self.device)
```

Our MLP is defined in Line 225 as follows.

```sh
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
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
```

6) CLIP backbone settings.

./utils/clip_util.py is the utils that CLIP will use. For FedAVG and LoRA, you have to do the following steps to ensure correct gradient update and parameter optimization (Line 28-30):
```sh
def freeze_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
```
For ours, FedCLIP, PromptFL, CocoOP abd LP++, you have to set it as False:
```sh
def freeze_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
```




















