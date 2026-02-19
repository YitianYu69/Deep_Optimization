import math
import torch
from torch import Tensor
import torch.nn.functional as F


import triton
import triton.language as tl

import numpy as np

from typing import Tuple, Union
from functools import reduce
import operator


def default_act_tensor_reshape(X: Tensor, Group_Size: int, act_padding: bool, pack_only: bool) -> Tuple[Tensor, int, int, Tuple[int, int, int, int], bool]:
    N = X.shape[0]
    _shape = X.shape

    channel_mean = False
    # if not pack_only and len(X.shape) == 4 and X.shape[1] > 2 and X.shape[1] % 2 == 0:
    #     channel_mean = True

    #     C = X.shape[1]
    #     _C = C // 2
    #     X = X.view(N, 2, _C, X.shape[2], X.shape[3])

    #     # N = X.shape[0]
    #     # _N = N // 2
    #     # N = _N
    #     # X = X.view(2, _N, X.shape[1], X.shape[2], X.shape[3])
    #     X = X.mean(dim=1)
    #     _shape = X.shape        

    if act_padding:
        flat = X.reshape(X.shape[0], -1)
        num_features = flat.size(1)
        pad_num = (-num_features) % Group_Size
        if pad_num > 0:
            flat = torch.cat(
                [flat, torch.zeros((X.shape[0], pad_num), dtype=flat.dtype, device=flat.device)], dim=1
            )
        X = flat

    x2 = X.reshape(N, -1, Group_Size)
    G = x2.shape[1]

    return x2, N, G, _shape, channel_mean



def default_act_tensor_reshape_back(X: Tensor, _shape: Tuple[int, int, int, int], act_padding: bool, channel_mean: bool) -> Tensor:
    N = _shape[0]

    if act_padding:
        num_features = reduce(operator.mul, _shape[1:], 1)
        flat = X.view(N, -1)[:, :num_features]
    else:
        flat = X.view(N, -1)

    if channel_mean:
        flat = flat.view(*_shape)
        flat = flat.repeat(1, 2, 1, 1)
        # flat = flat.repeat(2, 1, 1, 1)
    else:
        flat = flat.view(*_shape)
    return flat



def act_tensor_reshape_pad_hepler(X: Tensor, H: int, W: int, sgs: int) -> Tensor:

    pad_H = -(H) % sgs
    pad_W = -(W) % sgs

    if pad_H > 0 or pad_W > 0:
        X = F.pad(X, (0, pad_W, 0, pad_H), "constant", 0)
    return X, (X.shape[2] // sgs), (X.shape[3] // sgs)


def spatical_aware_act_tensor_reshape_eligibility(X: Tensor, Group_Size: int, act_padding: bool) -> Tuple[bool, bool]:
    if act_padding:
        return False, False
    elif len(X.shape) !=4:
        return False, False

    H = X.shape[2]
    W = X.shape[3]

    sub_group_size = int(np.sqrt(Group_Size))

    spatical_padding_eligibility = sub_group_size * sub_group_size != Group_Size
    spatical_reshape_eligibility = (H % sub_group_size == 0) and (W % sub_group_size == 0)

    return spatical_padding_eligibility, spatical_reshape_eligibility



def spatical_aware_act_tensor_reshape(X: Tensor, Group_Size: int, act_padding: bool, pack_only: bool) -> Tuple[Tensor, int, int, int, int, Tuple[int, int, int, int], bool, bool, bool]:
    spatical_padding_eligibility, spatical_reshape_eligibility = spatical_aware_act_tensor_reshape_eligibility(X, Group_Size, act_padding)

    channel_mean = False
    if not spatical_reshape_eligibility:
        x2, N, G, ori_shape, channel_mean = default_act_tensor_reshape(X, Group_Size, act_padding, pack_only)
        return x2, N, G, 0, 0, ori_shape, False, False, channel_mean
    
    N = X.shape[0]

    C = X.shape[1]
    H = X.shape[2]
    W = X.shape[3]

    _shape = X.shape
    sub_group_size = int(np.sqrt(Group_Size))
    new_H = H // sub_group_size
    new_W = W // sub_group_size

    if not spatical_padding_eligibility and spatical_reshape_eligibility:
        X, new_H, new_W = act_tensor_reshape_pad_hepler(X, H, W, sub_group_size)

    x2 = X.view(N, C, new_H, sub_group_size, new_W, sub_group_size)
    x2 = x2.permute(0,1,2,4,3,5) # N, C, H//sgs, W//sgs, sgs, sgs
    x2 = x2.reshape(N, C, new_H, new_W, Group_Size)
    x2 = x2.reshape(N, -1, Group_Size)

    G = x2.shape[1]
    return x2.contiguous(), N, G, new_H, new_W, _shape, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean


def spatical_aware_act_tensor_reshape_back(X: Tensor, _shape: Tuple[int, int, int, int], Group_Size: int, new_H: int, new_W: int, act_padding: bool, spatical_padding_eligibility: bool, spatical_reshape_eligibility: bool, channel_mean: bool) -> Tensor:

    if spatical_reshape_eligibility:
        N, C = _shape[0], _shape[1]
        sub_group_size = int(np.sqrt(Group_Size))

        x2 = X.view(N, C, new_H, new_W, Group_Size)
        x2 = x2.view(N, C, new_W, new_H, sub_group_size, sub_group_size)
        x2 = x2.permute(0,1,2,4,3,5) 
        x2 = x2.reshape(N, C, new_H * sub_group_size, new_W * sub_group_size)

        if not spatical_padding_eligibility:
            H = _shape[2]
            W = _shape[3]
            x2 = x2[:, :, :H, :W]

        if channel_mean:
            x2 = x2.view(*_shape)
            x2 = x2.repeat(1, 2, 1, 1)
        else:
            x2 = x2.view(*_shape)
        return x2.contiguous()
    else:
        return default_act_tensor_reshape_back(X, _shape, act_padding, channel_mean).contiguous()



