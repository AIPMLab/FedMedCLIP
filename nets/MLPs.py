import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout, Linear
from torch.nn.functional import normalize
from torch.nn import functional as F


class ImageMlp(nn.Module):
    def __init__(self):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 512)
        self.tanh = nn.Tanh()

    def _ff_block(self, x):
        # x = normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.fc2(self.dp(feat))
        out = self.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        # mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output


class TextMlp(nn.Module):
    def __init__(self):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 512)
        self.tanh = nn.Tanh()

    def _ff_block(self, x):
        x = normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.fc2(self.dp(feat))
        out = self.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        # mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output

class FC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FC, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
