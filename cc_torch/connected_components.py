import torch
from cc_torch import _C


def connected_components_labeling(x, relabel=False):
    """
    Connected Components Labeling by Block Union Find(BUF) algorithm.

    Args:
        x (cuda.ByteTensor): must be uint8, cuda and even num shapes
        relabel (bool): whether to return labels in range [0, max_label]

    Return:
        label (cuda.IntTensor)
    """
    if x.ndim == 2:
        assert x.shape[0] % 2 == 0 and x.shape[1] % 2 == 0
        ret = _C.cc_2d(x)
    elif x.ndim == 3:
        assert x.shape[0] % 2 == 0 and x.shape[1] % 2 == 0 and x.shape[2] % 2 == 0
        ret = _C.cc_3d(x)
    else:
        raise ValueError("x must be [H, W] or [D, H, W] shapes")
        
    if relabel:
        vs, idxs = torch.unique(ret, return_inverse=True, sorted=True)
        ret = torch.arange(len(vs), device=vs.device)[idxs]
        
    return ret
