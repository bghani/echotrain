"""Compatibility shim for legacy/models.py.

models.py does `from util import min_max_norm`. This is that one function,
copied from the old util.py without the rest of that module.
"""


def min_max_norm(x, min_val=-39.4655, max_val=53.6203):
    return (x - min_val) / (max_val - min_val)
