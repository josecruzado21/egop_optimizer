import random
from typing import Union
import torch
from collections import defaultdict

def stratified_split(
    dataset: Union[list, torch.Tensor],
    group1_perc: float = 0.5, 
    seed: int = 42
) -> tuple[list, list]:
    """
    Splits a dataset into two groups (e.g., dev and test) in a stratified manner, preserving class proportions.

    Args:
        dataset (list, torch.Tensor): Each item is either a tuple (data, label), or a label (int/tensor).
        group1_perc (float): Proportion of samples from each class to include in the first group.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[list, list]: Indices for group 1 and group 2.
    """
    random.seed(seed)
    class_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        if isinstance(item, (tuple, list)):
            label = item[1]
        else:
            label = item
        class_to_indices[label].append(idx)

    group1_indices, group2_indices = [], []
    for indices in class_to_indices.values():
        random.shuffle(indices)
        split = int(group1_perc * len(indices))
        group1_indices.extend(indices[:split])
        group2_indices.extend(indices[split:])
    return group1_indices, group2_indices