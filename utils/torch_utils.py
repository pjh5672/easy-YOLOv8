import time
import math
from copy import deepcopy

import torch
from torch import nn
from torch import optim
from thop import profile


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def model_info(model, input_size):
    mac, params = profile(deepcopy(model), 
                            inputs=(torch.randn(1, 3, input_size, input_size),), verbose=False)
    mb, gb = 1E+6, 1E+9
    print(f'Model Params(M): {params / mb:.2f}, FLOPs(G): {2 * mac / gb:.2f}')


class ModelEMA(nn.Module):
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        super(ModelEMA, self).__init__()
        self.module = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.module.parameters():
            p.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def update_parameters(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.state_dict()  # model state_dict
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
    

def build_optimizer(model, lr=0.001, momentum=0.9, weight_decay=1e-5):
    # No bias decay heuristic recommendation
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=False):
            if p_name == 'bias': # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # Norm's weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # Conv's weight (with decay)

    optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def build_scheduler(optimizer, cos_lr, lr_decay=1e-2, num_epochs=300):
    if cos_lr:
        lf = one_cycle(1, lr_decay, num_epochs)
    else:
        lf = lambda x: (1 - x / num_epochs) * (1.0 - lr_decay) + lr_decay
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lf)
    return scheduler, lf
