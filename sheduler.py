import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

thresholds = np.linspace(0.01, 0.99, 100)
# for i in range(0, 100):
#     print(thresholds[i])