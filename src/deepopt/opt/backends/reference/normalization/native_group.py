"""Materialized FP32 oracle for staged native-statistics GN, not GPU scratch."""
import torch

from deepopt.opt.nn.normalization.geometry import output_strides
from .activation import apply, adjust


def forward(x, weight, bias, contract, *, activation=None, preactivation=None, execution_info=None):
    source = x.to(contract.compute_dtype)
    gamma = weight.to(contract.compute_dtype) if contract.cast_affine and weight is not None else weight
    beta = bias.to(contract.compute_dtype) if contract.cast_affine and bias is not None else bias
    values = source.float().reshape(contract.domains, contract.elements)
    variance, mean = torch.var_mean(values, dim=1, correction=0)
    inverse = torch.rsqrt(variance + contract.eps)
    channel_shape = (contract.n, contract.groups, contract.channels // contract.groups)
    affine = torch.ones(contract.channels, device=x.device) if gamma is None else gamma.float()
    multiplier = inverse.reshape(contract.n, contract.groups, 1) * affine.reshape(1, contract.groups, -1)
    offset = -mean.reshape(contract.n, contract.groups, 1) * multiplier
    if beta is not None:
        offset = offset + beta.float().reshape(1, contract.groups, -1)
    output_values = (source.float().reshape(contract.n, contract.channels, contract.spatial) *
                     multiplier.reshape(channel_shape).reshape(contract.n, contract.channels, 1) +
                     offset.reshape(contract.n, contract.channels, 1)).reshape(contract.shape)
    output = torch.empty_strided(contract.shape, output_strides(source), device=x.device, dtype=contract.compute_dtype)
    output.copy_(output_values)
    if activation:
        output.copy_(apply(output, activation, preactivation))
    if execution_info is not None:
        execution_info.update(contract.receipt(), semantic_reference_fp32_temporaries=True,
                              native_group_staged=True, forward_global_fp32_staging_output=True)
    return output, mean.to(contract.statistics_dtype), inverse.to(contract.statistics_dtype), source, gamma


def backward(gradient, source, mean, inverse, weight, contract, *, needs=(True, True, True),
             derivative_input=None, derivative_activation=None, execution_info=None):
    gradient = adjust(gradient, derivative_input, derivative_activation)
    dy = gradient.float().reshape(contract.n, contract.channels, contract.spatial)
    if not (needs[0] or needs[1]):
        return None, None, dy.sum((0, 2)).to(contract.affine_dtype) if needs[2] else None
    x = source.float().reshape_as(dy)
    mu = mean.float().reshape(contract.n, contract.groups, 1)
    inv = inverse.float().reshape_as(mu)
    ds, db = (dy * x).sum(2), dy.sum(2)
    ds_group, db_group = ds.reshape(contract.n, contract.groups, -1), db.reshape(contract.n, contract.groups, -1)
    dw = ((ds_group - db_group * mu) * inv).reshape(contract.n, contract.channels).sum(0) if needs[1] else None
    dbias = db.sum(0) if needs[2] else None
    dx = None
    if needs[0]:
        gamma = torch.ones(contract.channels, device=gradient.device) if weight is None else weight.float()
        gamma = gamma.reshape(1, contract.groups, -1)
        ds_weighted, db_weighted = (ds_group * gamma).sum(2, keepdim=True), (db_group * gamma).sum(2, keepdim=True)
        c2 = (db_weighted * mu - ds_weighted) * inv * inv * inv / contract.elements
        c3 = -c2 * mu - db_weighted * inv / contract.elements
        c1 = (inv * gamma).reshape(contract.n, contract.channels, 1)
        c2 = c2.expand(-1, -1, contract.channels // contract.groups).reshape_as(c1)
        c3 = c3.expand(-1, -1, contract.channels // contract.groups).reshape_as(c1)
        dx = (dy * c1 + x * c2 + c3).reshape(contract.shape).to(contract.input_dtype)
    if execution_info is not None:
        execution_info.update(contract.receipt(), semantic_reference_fp32_temporaries=True,
                              native_group_staged=True, activation_derivative_fused=derivative_activation is not None)
    return dx, dw.to(contract.affine_dtype) if dw is not None else None, dbias.to(contract.affine_dtype) if dbias is not None else None
