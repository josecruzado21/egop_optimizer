import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

class Conv2d_reparam(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, V, 
                 stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        self.V = V
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        V = self.V
        W = self.weight
        W_new = torch.einsum('ij,bjk->bik', V.to(W.device),  W.view(W.shape[0], -1).unsqueeze(-1)).view(W.shape)
        return F.conv2d(input, W_new, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        groups=self.groups)
    
    def recover_original_weights(self):
        with torch.no_grad():
            V = self.V
            W = self.weight
            W_orig = torch.einsum('ij,bjk->bik', V.to(W.device),  W.view(W.shape[0], -1).unsqueeze(-1)).view(W.shape)
        return W_orig

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            w_flat = self.weight.view(self.weight.shape[0], -1)
            self.V = self.V.to(self.weight.device)
            w_new = torch.einsum('ij,bj->bi', self.V.T, w_flat)
            self.weight.copy_(w_new.view_as(self.weight))





class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, V1=None, V2=None):
        super().__init__()
        padding = kernel_size // 2
        self.stride = stride
        self.out_channels = out_channels
        if (V1 is not None) and (V2 is not None):
            self.conv1 = Conv2d_reparam(in_channels, out_channels, kernel_size, V1,
                                stride=stride, padding=padding, bias=False)
            self.conv2 = Conv2d_reparam(out_channels, out_channels, kernel_size, V2,
                                stride=1, padding=padding, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                                stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if out.shape != identity.shape:
            identity = identity_downsample(identity, self.out_channels, self.stride)

        out += identity
        out = self.relu(out)
        return out


def identity_downsample(x, out_channels, stride):
    if stride > 1:
        x = x[:, :, ::stride, ::stride]
    in_channels = x.size(1)
    if out_channels > in_channels:
        pad_channels = out_channels - in_channels
        x = F.pad(x, (0, 0, 0, 0, 0, pad_channels))
    return x