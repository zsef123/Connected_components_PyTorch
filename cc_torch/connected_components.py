import torch
from cc_torch import _C


def connected_components_labeling(x):
    """
    Connected Components Labeling by Block Union Find(BUF) algorithm.

    Args:
        x (cuda.ByteTensor): must be uint8, cuda and even num shapes

    Return:
        label (cuda.IntTensor)
    """
    if x.ndim == 2:
        return _C.cc_2d(x)
    elif x.ndim == 3:
        return _C.cc_3d(x)
    else:
        raise ValueError("x must be [H, W] or [D, H, W] shapes")
