"""Compatibility entry point for the former scale-preserving FP8 relay."""

import torch


def relu(operand):
    from ...activation.relu import forward
    from deepopt.opt.nn.activation.geometry import ReLUGeometry
    from deepopt.opt.quantization.activation_tensor import FP8Activation

    x = FP8Activation(operand.data, operand.scale_inv, tuple(operand.data.shape), operand.block_shape,
                      "last", torch.bfloat16)
    return forward(x, ReLUGeometry(tuple(x.shape)), dtype=operand.data.dtype,
                   block_shape=operand.block_shape, packed=False, save_mask=False, inplace=False, reuse=True)[0]
