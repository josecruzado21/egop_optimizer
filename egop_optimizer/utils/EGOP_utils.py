import torch
import itertools
from tqdm import tqdm
import torch.nn as nn
import time
import math

from egop_optimizer.utils.device_utils import get_available_device

import pdb

DEVICE = get_available_device()


def compute_k_gradients_linear_module(
    model_OG,
    module_name,
    data_loader,
    criterion,
    k,
    device=DEVICE,
    recalculate_V=False,
    current_model=None,
):
    infinite_loader = itertools.cycle(data_loader)
    model_OG = model_OG.to(device)
    module_OG = dict(model_OG.named_modules())[module_name]
    if recalculate_V:
        current_state_dict = {
            k: (
                module.recover_original_weights().detach().clone()
                if hasattr(module, "recover_original_weights")
                else v.detach().clone()
            )
            for k, v in current_model.state_dict().items()
            for name, module in current_model.named_modules()
            if name in k and k in model_OG.state_dict()
        }
        model_OG.load_state_dict(current_state_dict, strict=False)
        model_OG.zero_grad()
        original_weights_module = module_OG.weight.data.clone()
        del current_state_dict
    gradients = []
    for _ in tqdm(range(k), leave=False):
        if recalculate_V:
            perturbation = torch.rand_like(module_OG.weight)
            module_OG.weight.data += perturbation
        else:
            model_OG.reinitialize()
        model_OG.zero_grad()
        x, y = next(infinite_loader)
        x = x.to(device)
        y = y.to(device)
        output = model_OG(x)
        loss = criterion(output, y)
        loss.backward()
        grad = module_OG.weight.grad.detach().clone().flatten().to("cpu")
        gradients.append(grad)
        if recalculate_V:
            module_OG.weight.data = original_weights_module.clone()
    return torch.stack(gradients, dim=0).T


def compute_k_gradients_cnn_module(
    model_OG,
    module_name,
    data_loader,
    criterion,
    k,
    device=DEVICE,
    recalculate_V=False,
    current_model=None,
):
    infinite_loader = itertools.cycle(data_loader)
    model_OG = model_OG.to(device)
    module_OG = dict(model_OG.named_modules())[module_name]
    if recalculate_V:
        current_state_dict = {
            k: (
                module.recover_original_weights().detach().clone()
                if hasattr(module, "recover_original_weights")
                else v.detach().clone()
            )
            for k, v in current_model.state_dict().items()
            for name, module in current_model.named_modules()
            if name in k and k in model_OG.state_dict()
        }
        model_OG.load_state_dict(current_state_dict, strict=False)
        model_OG.zero_grad()
        original_weights_module = module_OG.weight.data.clone()
        del current_state_dict
    n_filters = module_OG.weight.shape[0]
    gradients = []
    k = math.ceil(k / module_OG.out_channels)
    for _ in tqdm(range(k), leave=False):
        if recalculate_V:
            perturbation = torch.empty_like(module_OG.weight)
            torch.nn.init.kaiming_normal_(
                perturbation, mode="fan_in", nonlinearity="relu"
            )
            module_OG.weight.data += perturbation
        else:
            model_OG.reinitialize()
        model_OG.zero_grad()
        x, y = next(infinite_loader)
        x = x.to(device)
        y = y.to(device)
        output = model_OG(x)
        loss = criterion(output, y)
        loss.backward()
        grad = module_OG.weight.grad.detach().clone().flatten().to("cpu")
        gradients.append(grad.view(n_filters, -1))
        if recalculate_V:
            module_OG.weight.data = original_weights_module.clone()
    gradients_tensor = torch.stack(gradients, dim=2)
    n_params_per_kernel = gradients_tensor.shape[1]
    return gradients_tensor.permute(1, 0, 2).reshape(n_params_per_kernel, -1)


def compute_V(gradients, use_randomized_svd=False, n_components=None):
    if use_randomized_svd:
        if n_components is None:
            # otherwise default = 6, which can lead to overly low-rank approximations
            n_components = min(list(gradients.shape))
        U, _, _ = torch.svd_lowrank(gradients, q=n_components)
    else:
        U, S, V = torch.svd(gradients, some=False, compute_uv=True)
        U, S, V = U.cpu(), S.cpu(), V.cpu()
    return U


def compute_V_by_layer_per_layer(
    model_OG,
    k,
    data_loader,
    criterion,
    device=DEVICE,
    reparam_linear_layers=False,
    n_components=None,
    recalculate_V=False,
    verbose=True,
    current_model=None,
):
    model_OG = model_OG.to(device)
    current_model = current_model.to(device) if current_model is not None else None
    EGOP_eigenbasis = dict()
    start_time_total = time.time()
    if not verbose:
        print("-" * 100)
        print("Computing V by layer:")
        print("-" * 100)
    for name, module in model_OG.named_modules():
        if isinstance(module, nn.Conv2d):
            if verbose:
                print()
                print("-" * 100)
                print(f"Computing V for layer Convolutional layer {name}:")
                print("-" * 100)
                print(f"Computing {k} gradients:")
            start_time_gradients = time.time()
            gradients_cnn = compute_k_gradients_cnn_module(
                model_OG,
                name,
                data_loader,
                criterion,
                k,
                device,
                recalculate_V=recalculate_V,
                current_model=current_model,
            )
            if verbose:
                print(
                    f"Gradients computed in {time.time() - start_time_gradients:.2f} seconds"
                )
            start_time_eigendecomposition = time.time()
            if device == "cuda":
                gradients_cnn = gradients_cnn.to(device)
            EGOP_eigenbasis[name] = compute_V(gradients_cnn)
            if verbose:
                print(
                    f"Computed V with shape {EGOP_eigenbasis[name].shape} in {time.time() - start_time_eigendecomposition:.2f} seconds"
                )
            del gradients_cnn
        elif isinstance(module, nn.Linear) and reparam_linear_layers:
            if verbose:
                print()
                print("-" * 100)
                print(f"Computing V for layer Linear layer {name}:")
                print("-" * 100)
                print(f"Computing {k} gradients:")
            start_time_gradients = time.time()
            k_linear = k // 10
            gradients_linear = compute_k_gradients_linear_module(
                model_OG,
                name,
                data_loader,
                criterion,
                k_linear,
                device,
                recalculate_V=recalculate_V,
                current_model=current_model,
            )
            if verbose:
                print(
                    f"Gradients computed in {time.time() - start_time_gradients:.2f} seconds"
                )
            start_time_eigendecomposition = time.time()
            if device == "cuda":
                gradients_linear = gradients_linear.to(device)
            EGOP_eigenbasis[name] = compute_V(
                gradients_linear, use_randomized_svd=True, n_components=n_components
            )
            if verbose:
                print(
                    f"Computed V with shape {EGOP_eigenbasis[name].shape} in {time.time() - start_time_eigendecomposition:.2f} seconds"
                )
            del gradients_linear
    end_time_total = time.time()
    print(
        "Total time to compute V by layer: {:.2f} seconds".format(
            end_time_total - start_time_total
        )
    )
    return EGOP_eigenbasis


def compute_k_gradients_all_layers(
    model_OG,
    data_loader,
    k,
    device,
    criterion,
    recalculate_V=False,
    current_model=None,
    reparam_linear_layers=False,
):
    print("-" * 100)
    print("Computing V by layer:")
    print("-" * 100)
    infinite_loader = itertools.cycle(data_loader)
    print("Finished preparing infinite loader")
    current_model = current_model.to(device) if current_model is not None else None
    model_OG = model_OG.to(device)
    if recalculate_V:
        current_state_dict = {
            k: (
                module.recover_original_weights().detach().clone()
                if hasattr(module, "recover_original_weights")
                else v.detach().clone()
            )
            for k, v in current_model.state_dict().items()
            for name, module in current_model.named_modules()
            if name in k and k in model_OG.state_dict()
        }
        model_OG.load_state_dict(current_state_dict, strict=False)
        model_OG.zero_grad()
        original_weights = {
            name: module.weight.data.clone()
            for name, module in model_OG.named_modules()
            if hasattr(module, "weight")
            and (module.weight is not None)
            and (isinstance(module, (nn.Conv2d, nn.Linear)))
        }
        del current_state_dict
    # Max number of out channels in the network
    min_out_channels = float("inf")
    for name, module in model_OG.named_modules():
        if isinstance(module, nn.Conv2d):
            min_out_channels = min(min_out_channels, module.out_channels)
    if min_out_channels == float("inf"):
        # If no convolutional layers, just use k as provided
        pass
    else:
        k = math.ceil(k / min_out_channels)
    gradients_dict = {}
    print("using k =", k, "for all layers")
    for _ in tqdm(range(k), leave=False):
        if recalculate_V:
            for name, module in model_OG.named_modules():
                if name in original_weights:
                    perturbation = torch.empty_like(module.weight)
                    torch.nn.init.kaiming_normal_(
                        perturbation, mode="fan_in", nonlinearity="relu"
                    )
                    module.weight.data += perturbation
        else:
            model_OG.reinitialize()
        model_OG.zero_grad()
        x, y = next(infinite_loader)
        x = x.to(device)
        y = y.to(device)
        output = model_OG(x)
        loss = criterion(output, y)
        loss.backward()

        for name, module in model_OG.named_modules():
            if hasattr(module, "weight") and (module.weight.grad is not None):
                if (isinstance(module, nn.Linear)) and (reparam_linear_layers):
                    grad = module.weight.grad.detach().clone()
                    if name not in gradients_dict:
                        gradients_dict[name] = []
                    gradients_dict[name].append(grad.flatten())
                elif isinstance(module, nn.Conv2d):
                    grad = module.weight.grad.detach().clone()
                    n_filters = module.weight.shape[0]
                    grad = grad.view(n_filters, -1)
                    if name not in gradients_dict:
                        gradients_dict[name] = []
                    gradients_dict[name].append(grad)
            if recalculate_V:
                if name in original_weights:
                    module.weight.data = original_weights[name].clone()

    # Stack gradients for each layer
    for name in gradients_dict:
        gradients_dict[name] = torch.stack(gradients_dict[name], dim=0)
        if gradients_dict[name].dim() == 3:
            n_params_per_kernel = gradients_dict[name].shape[2]
            gradients_dict[name] = (
                gradients_dict[name].permute(2, 1, 0).reshape(n_params_per_kernel, -1)
            )
        else:
            gradients_dict[name] = gradients_dict[name].reshape(
                gradients_dict[name].shape[1], -1
            )
    return gradients_dict


def compute_V_by_layer(
    model_OG,
    k,
    data_loader,
    criterion,
    device=DEVICE,
    reparam_linear_layers=True,
    use_randomized_svd=False,
    n_components=None,
    recalculate_V=False,
    current_model=None,
    per_layer=False,
    verbose=False,
):
    if per_layer:
        raise Exception("Option per_layer not yet supported.")
        # compute_V_by_layer_per_layer(
        #     model_OG=model_OG,
        #     k=k,
        #     data_loader=data_loader,
        #     criterion=criterion,
        #     device=device,
        #     reparam_linear_layers=reparam_linear_layers,
        #     n_components=n_components,
        #     recalculate_V=recalculate_V,
        #     verbose=verbose,
        #     current_model=current_model,
        # )
    else:
        beg = time.time()
        gradients = compute_k_gradients_all_layers(
            model_OG=model_OG,
            data_loader=data_loader,
            k=k,
            device=device,
            recalculate_V=recalculate_V,
            current_model=current_model,
            reparam_linear_layers=reparam_linear_layers,
            criterion=criterion,
        )
        V_dict = {}
        for name, module in model_OG.named_modules():
            if name in gradients:
                grad = gradients[name]
                if device == "cuda":
                    grad = grad.to(device)
                else:
                    grad = grad.to("cpu")
                if isinstance(module, nn.Linear) and reparam_linear_layers:
                    V_dict[name] = compute_V(
                        grad,
                        use_randomized_svd=use_randomized_svd,
                        n_components=n_components,
                    )
                elif isinstance(module, nn.Conv2d):
                    V_dict[name] = compute_V(grad)
        end = time.time()
        print()
        print(
            "Total time to compute V by layer (all layers together): {:.2f}".format(
                end - beg
            )
        )
        print()
        return V_dict
