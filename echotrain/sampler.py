"""Class-balanced sampling for imbalanced training sets."""
from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


def balanced_sampler(labels, cap=1000, power=0.5, noise_class=None, noise_multiplier=0.4):
    """One draw per training file per epoch, with replacement.

    A file of class c is drawn with weight 1 / min(count_c, cap) ** power.
    power=1 balances classes fully, power=0 keeps the natural distribution,
    0.5 is a compromise. `noise_class` (if given) is further down-weighted.
    """
    counts = Counter(labels)
    class_w = {c: 1.0 / (min(n, cap) ** power) for c, n in counts.items()}
    if noise_class is not None:
        if noise_class not in class_w:
            raise ValueError(f"sampler.noise_class {noise_class!r} is not a training class")
        class_w[noise_class] *= noise_multiplier
    weights = torch.DoubleTensor([class_w[c] for c in labels])
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
