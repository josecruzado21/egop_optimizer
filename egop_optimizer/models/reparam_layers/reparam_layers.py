import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

from typing import Optional, Union

from egop_optimizer.utils.device_utils import get_available_device

DEVICE = get_available_device()


class Conv2d_reparam(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        V,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
    ):
        self.V = V
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        V = self.V
        W = self.weight
        W_new = torch.einsum(
            "ij,bjk->bik", V.to(W.device), W.view(W.shape[0], -1).unsqueeze(-1)
        ).view(W.shape)
        return F.conv2d(
            input,
            W_new,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def recover_original_weights(self):
        with torch.no_grad():
            V = self.V
            W = self.weight
            W_orig = torch.einsum(
                "ij,bjk->bik", V.to(W.device), W.view(W.shape[0], -1).unsqueeze(-1)
            ).view(W.shape)
        return W_orig

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            w_flat = self.weight.view(self.weight.shape[0], -1)
            self.V = self.V.to(self.weight.device)
            w_new = torch.einsum("ij,bj->bi", self.V.T, w_flat)
            self.weight.copy_(w_new.view_as(self.weight))


class ResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, V1=None, V2=None
    ):
        super().__init__()
        padding = kernel_size // 2
        self.stride = stride
        self.out_channels = out_channels
        if (V1 is not None) and (V2 is not None):
            self.conv1 = Conv2d_reparam(
                in_channels,
                out_channels,
                kernel_size,
                V1,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv2 = Conv2d_reparam(
                out_channels,
                out_channels,
                kernel_size,
                V2,
                stride=1,
                padding=padding,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            )
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


def _convert_V_to_tensor(V, device):
    """Convert V matrix from numpy or tensor to a float tensor on the given device."""
    if type(V) == np.ndarray:
        return torch.from_numpy(V).type(torch.FloatTensor).to(device)
    elif type(V) == torch.Tensor:
        return V.to(device)
    else:
        raise Exception(
            "Unsupported format for reparameterization matrix. Expected np.ndarray or torch.Tensor."
        )


class EGOP_linear_layer(torch.nn.Module):
    """
    A custom linear layer that reparameterizes the weight matrix in the EGOP eigenbasis.

    This layer modifies the standard PyTorch `Linear` layer to allow reparameterization
    of weights using a provided eigenbasis matrix `V` (and optionally its inverse `V_inv`).
    This transformation is applied independently for each layer using its own eigenbasis.

    Args:
        V (Union[np.ndarray, torch.Tensor]): A square matrix of shape
            `(in_features * out_features, in_features * out_features)` whose columns are
            the eigenvectors for reparameterization.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        V_inv (Optional[Union[np.ndarray, torch.Tensor]]): Optional inverse of `V`. If `None`,
            assumes `V` is orthonormal and uses `V.T` as the inverse.
        bias (bool): If True, includes a learnable bias parameter.

    Raises:
        Exception: If `V` is not a NumPy array or Tensor, or if `V` is not orthonormal
            and no valid inverse is provided.
    """

    def __init__(
        self,
        V: Union[np.ndarray, torch.Tensor],
        in_features: int,
        out_features: int,
        V_inv: Optional[Union[np.ndarray, torch.Tensor]] = None,
        bias: bool = False,
        use_approximately_orthogonal_matrix: bool = True,
        device=DEVICE,
        verbose=False,
    ):
        """
        Input matrix V should be a square matrix of side length dim_1 * dim_2, the eigenbasis for the layer
        Columns of V should be eigenvectors.
        """
        super().__init__()
        # Pytorch applies weights W to input x as xW.T, hence W is out x in
        # Matching shape conventions of torch.nn.Linear, weights are out x in
        self.weight = torch.nn.parameter.Parameter(
            torch.empty((out_features, in_features), device=device)
        )
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.empty(out_features, device=device)
            )
        else:
            self.register_parameter("bias", None)

        # V should be a (in_features*out_features) x (in_features*out_features) matrix
        if type(V) == np.ndarray:
            V = torch.from_numpy(V).type(torch.FloatTensor).to(device)
        elif type(V) == torch.Tensor:
            V = V.to(device)
        else:
            raise Exception(
                "Linear layers: Unsupported format for reparameterization matrix."
            )
        self.V = V
        if verbose:
            print(
                f"V matrix of OG: {V.shape} | {V.numel():,} elements | {V.numel() * V.element_size() / 1024 / 1024:.2f} MB"
            )
        if V_inv is None:
            V_inv = V.T
        if type(V_inv) == np.ndarray:
            V_inv = torch.from_numpy(V_inv).type(torch.FloatTensor).to(device)
        elif type(V_inv) == torch.Tensor:
            V_inv = V_inv.to(device)
        if use_approximately_orthogonal_matrix:
            print(
                f"ORTHOGONALITY CHECK DISABLED because use_approximately_orthogonal_matrix set to True."
            )
        else:
            try:
                num_params = in_features * out_features
                # Generous bar for inverse status
                assert torch.allclose(
                    torch.eye(num_params, device=device), V @ V_inv, atol=1e-3
                )
            except:
                raise Exception(
                    "Linear layers: No inverse provided, but V is not orthonormal."
                )
        self.V_inv = V_inv

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        """
        Apply the linear transformation using reparameterized weights in the EGOP eigenbasis.

        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.

        Args:
            input (torch.Tensor): Input tensor of shape `(batch_size, in_features)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, out_features)`.
        """
        W_prime = torch.reshape(
            torch.matmul(self.V, self.weight.flatten()),
            shape=(self.out_features, self.in_features),
        )
        return F.linear(input, W_prime, bias=self.bias)

    def return_weight_copy_in_OG_coors(self):
        # Detach and clone weights so that edits do not affect upstream weights
        W_prime = torch.reshape(
            torch.matmul(self.V, self.weight.detach().clone().flatten()),
            shape=(self.out_features, self.in_features),
        )
        return W_prime


class EGOP_auxiliary_variables_linear_layer(torch.nn.Module):
    """
    A custom linear layer that reparameterizes the weight matrix in the EGOP eigenbasis
    using auxiliary variables.

    This layer uses V of size d x r, but does NOT restrict the optimizer to working in a
    reduced space because it introduces auxiliary variables that live in range(V^perp),
    the orthogonal complement to range(V).

    The forward pass computes:
        W = weight_d + V @ (weight_r - V^T @ weight_d)

    This is mathematically equivalent to:
        W = V @ weight_r + (I - V @ V^T) @ weight_d

    Args:
        V (Union[np.ndarray, torch.Tensor]): A d x r matrix whose columns are
            the eigenvectors for reparameterization.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): If True, includes a learnable bias parameter.
    """

    def __init__(
        self,
        V: Union[np.ndarray, torch.Tensor],
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=DEVICE,
    ):
        super().__init__()
        self.d = in_features * out_features
        self.r = V.shape[1]
        if self.r >= self.d:
            import warnings

            warnings.warn(
                "EGOP_auxiliary_variables_linear_layer created, but provided matrix is of shape d x r for r >= d."
            )
        # weight_r parameters are of dimension r, weight_d are of dimension d
        self.weight_r = torch.nn.parameter.Parameter(
            torch.empty((self.r), device=device)
        )
        self.weight_d = torch.nn.parameter.Parameter(
            torch.empty((self.d), device=device)
        )
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.empty(out_features, device=device)
            )
        else:
            self.register_parameter("bias", None)

        self.V = _convert_V_to_tensor(V, device)

        # Check that V has orthonormal columns
        try:
            assert torch.all(
                torch.isclose(
                    self.V.T @ self.V,
                    torch.eye(self.r, device=device),
                    atol=1e-5,
                )
            )
        except:
            raise Exception(
                "Provided matrix V does not have orthonormal columns to specified tolerance."
            )

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        """
        Compute W = weight_d + V @ (weight_r - V^T @ weight_d), then apply as linear layer.

        This uses 2 matmul ops instead of 3 by leveraging:
            V @ w_r + (I - V V^T) w_d = w_d + V @ (w_r - V^T w_d)
        """
        W_prime = torch.reshape(
            self.weight_d + self.V @ (self.weight_r - self.V.T @ self.weight_d),
            shape=(self.out_features, self.in_features),
        )
        return F.linear(input, W_prime, bias=self.bias)

    def return_weight_copy_in_OG_coors(self):
        # Detach and clone weights so that edits do not affect upstream weights
        W_prime = torch.reshape(
            self.weight_d + self.V @ (self.weight_r - self.V.T @ self.weight_d),
            shape=(self.out_features, self.in_features),
        )
        return W_prime
