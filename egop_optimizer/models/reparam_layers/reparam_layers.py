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

class Linear_reparam(nn.Linear):
    def __init__(self, in_features, out_features, V, bias=False):
        super().__init__(in_features, out_features, bias)
        self.V = V

    def forward(self, input):
        V = self.V
        W = self.weight
        W_new = (V.to(W.device) @ W.flatten()).view(W.shape)
        return F.linear(input, W_new, self.bias)
    
    def recover_original_weights(self):
        with torch.no_grad():
            V = self.V
            W = self.weight
            W_orig = (V.to(W.device) @ W.flatten()).view(W.shape)
        return W_orig    

class EGOP_auxiliary_variables_linear_layer(nn.Linear):
    def __init__(
        self,
        V,
        in_features,
        out_features,
        bias=True,
    ):
        """
        Input matrix V should be a matrix of side length dim_1 * dim_2 x r, the r leading eigenvectors for this layer's EGOP.
        """
        if type(V) == np.ndarray:
            V = torch.from_numpy(V).type(torch.FloatTensor)
        elif type(V) == torch.Tensor:
            V = V
        else:
            raise Exception(
                "Linear layers: Unsupported format for reparameterization matrix."
            )
        self.V = V
        self.d = in_features * out_features
        self.r = V.shape[1]
        super().__init__(in_features, out_features, bias)
        self.weight_r = torch.nn.parameter.Parameter(torch.empty((self.r), device=self.weight.device))
        self.weight_d = torch.nn.parameter.Parameter(torch.empty((self.d), device=self.weight.device))
        self.reset_parameters()
        try:
            assert torch.all(torch.isclose(V.T @ V, torch.eye(self.r).to(V.device), atol=1e-5))
        except:
            raise Exception(
                "Provided matrix V does not have orthonormal columns to specified tolerance."
            )

    def forward(self, input):
        V = self.V
        V = V.to(self.weight_r.device)
        W_prime = torch.reshape(
            self.weight_d + V @ (self.weight_r - V.T @ self.weight_d),
            shape=(self.out_features, self.in_features),
        )
        return F.linear(input, W_prime, bias=self.bias)
    
    def recover_original_weights(self):
        with torch.no_grad():
            V = self.V
            weight_r = self.weight_r
            weight_d = self.weight_d
            W = torch.reshape(weight_d + V.to(weight_d.device) @ (weight_r - V.to(weight_d.device).T @ weight_d),
                              shape=(self.out_features, self.in_features))
        return W    
    
    def reset_parameters(self):
        # If auxiliary parameters are not yet created (called from parent __init__), only run base init.
        if not (hasattr(self, "weight_r") and hasattr(self, "weight_d") and hasattr(self, "V")):
            return super().reset_parameters()

        # full initialization when aux params exist
        super().reset_parameters()
        with torch.no_grad():
            w_flat = self.weight.view(-1)  # Flatten weight
            V = self.V.to(self.weight_r.device)

            # Ensure dimensions match for matrix multiplication
            if V.shape[0] != w_flat.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: V has shape {V.shape}, but w_flat has shape {w_flat.shape}."
                )

            theta_r = torch.matmul(V.T, w_flat)
            VVT_w = torch.matmul(V, torch.matmul(V.T, w_flat))
            theta_d = w_flat - VVT_w
            self.weight_r.copy_(theta_r.view_as(self.weight_r))
            self.weight_d.copy_(theta_d.view_as(self.weight_d))

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