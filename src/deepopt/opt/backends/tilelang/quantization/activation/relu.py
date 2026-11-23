"""Compatibility FP8 ReLU relay."""


def relu(operand):
    import torch
    from deepopt.opt.nn.activation.geometry import ReLUGeometry
    from deepopt.opt.quantization.activation_tensor import FP8Activation
    from ...activation.relu import forward

    value = FP8Activation(operand.data, operand.scale_inv, tuple(operand.data.shape), operand.block_shape,
                          "last", torch.bfloat16)
    return forward(value, ReLUGeometry(tuple(value.shape)), dtype=operand.data.dtype,
                   block_shape=operand.block_shape, packed=False, save_mask=False, inplace=False, reuse=True)[0]
