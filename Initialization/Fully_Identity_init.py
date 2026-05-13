import torch
from torch import nn


# ------------------------------------------------------------
# Core paper-style identity-like matrix
# ------------------------------------------------------------
@torch.no_grad()
def build_idinit_matrix(
    out_dim: int,
    in_dim: int,
    *,
    loosen_eps: float = 1e-6,
    alpha: float = 1.0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Build a padded identity-like matrix for non-square shapes.

    This is a practical implementation of the paper's idea:
    - do NOT zero-pad a partial identity only once
    - instead, pad a new identity adjacent to an identity matrix,
      repeating identity blocks until rows are filled

    Shape: [out_dim, in_dim]

    Example:
      out_dim=10, in_dim=4

      rows 0:4   <- I_4
      rows 4:8   <- I_4
      rows 8:10  <- I_2

    A tiny noise term is added as the loosening condition.
    """
    W = torch.zeros((out_dim, in_dim), device=device, dtype=dtype)

    r = 0
    while r < out_dim:
        block = min(in_dim, out_dim - r)
        W[r:r + block, :block] = torch.eye(block, device=device, dtype=dtype)
        r += block

    W.mul_(alpha)

    if loosen_eps > 0:
        W.add_(torch.randn_like(W) * loosen_eps)

    return W


# ------------------------------------------------------------
# Linear / Embedding style
# ------------------------------------------------------------
@torch.no_grad()
def idinit_linear_(
    layer: nn.Linear,
    *,
    loosen_eps: float = 1e-6,
    alpha: float = 1.0,
    bias_zero: bool = True,
) -> nn.Linear:
    W = build_idinit_matrix(
        layer.out_features,
        layer.in_features,
        loosen_eps=loosen_eps,
        alpha=alpha,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    layer.weight.copy_(W)

    if layer.bias is not None and bias_zero:
        layer.bias.zero_()

    return layer


# ------------------------------------------------------------
# Conv2d: patch-maintain version
# ------------------------------------------------------------
@torch.no_grad()
def idinit_conv2d_patch_(
    layer: nn.Conv2d,
    *,
    loosen_eps: float = 1e-6,
    alpha: float = 1.0,
    bias_zero: bool = True,
    prior_data=None,
    freq_ratio=0.05
) -> nn.Conv2d:
    """
    Patch-maintain convolution initialization.

    The paper warns that plain channel-maintain identity can cause severe
    degeneration. Their fix is to reshape an identity-like matrix into the
    conv tensor, which shifts spatial features and increases channel diversity.

    We implement that as:
        weight shape = [out_c, in_c, kH, kW]
        flatten input patch dimension to in_c * kH * kW
        initialize a matrix with IDInit
        reshape back to conv tensor
    """
    out_c, in_c, kH, kW = layer.weight.shape

    flat_in = in_c * kH * kW
    W2 = build_idinit_matrix(
        out_c,
        flat_in,
        loosen_eps=loosen_eps,
        alpha=alpha,
        device=layer.weight.device,
        dtype=layer.weight.dtype,

    )
    W4 = W2.view(out_c, in_c, kH, kW)

    if prior_data is not None:
        freq = prior_data.to(W4.dtype).to(W4.device)

        ch, cw = kH // 2, kW // 2
        mask = torch.ones_like(W4)
        mask[:, :, ch, cw] = 0.0

        freq = freq / (freq.std(unbiased=False) + 1e-8)

        W4 = W4 + mask * freq * freq_ratio

    layer.weight.copy_(W4)

    if layer.bias is not None and bias_zero:
        layer.bias.zero_()

    return layer


# ------------------------------------------------------------
# Optional reference: channel-maintain fallback
# Not recommended by the paper for performance, but useful for ablation.
# ------------------------------------------------------------
@torch.no_grad()
def idinit_conv2d_channel_(
    layer: nn.Conv2d,
    *,
    loosen_eps: float = 1e-6,
    alpha: float = 1.0,
    bias_zero: bool = True,
) -> nn.Conv2d:
    out_c, in_c, kH, kW = layer.weight.shape
    W = torch.zeros_like(layer.weight)
    ch = kH // 2
    cw = kW // 2

    r = 0
    while r < out_c:
        block = min(in_c, out_c - r)
        W[r:r + block, :block, ch, cw] = torch.eye(
            block, device=W.device, dtype=W.dtype
        )
        r += block

    W.mul_(alpha)

    if loosen_eps > 0:
        W.add_(torch.randn_like(W) * loosen_eps)

    layer.weight.copy_(W)

    if layer.bias is not None and bias_zero:
        layer.bias.zero_()

    return layer


@torch.no_grad()
def build_idiz_matrix(
    out_dim: int,
    in_dim: int,
    *,
    eps: float = 1e-6,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Paper-faithful zero-preserving matrix IDIZ_eps.

    Key idea:
      - start from IDI_eps
      - add a matched negative-eps pattern so outputs stay near zero
        while avoiding dead neurons

    Cases from the paper:
      1) out_dim < in_dim:
         W[:, out_dim:] = IDI_{-eps} over the remaining columns
      2) out_dim >= in_dim:
         add a cyclic shifted -eps entry in each row

    This keeps the residual branch's last transform near-zero, but trainable.
    """
    if out_dim <= 0 or in_dim <= 0:
        raise ValueError("out_dim and in_dim must be positive")

    # Start from IDI_eps
    W = build_idinit_matrix(
        out_dim,
        in_dim,
        loosen_eps=0.0,   # no loosen noise for IDIZ
        alpha=eps,
        device=device,
        dtype=dtype,
    )

    # Edge case: one input channel/feature only
    # Can't form x_j - x_{j+1}, so alternate signs across rows.
    if in_dim == 1:
        signs = torch.where(
            (torch.arange(out_dim, device=device) % 2) == 0,
            torch.ones(out_dim, device=device, dtype=dtype),
            -torch.ones(out_dim, device=device, dtype=dtype),
        )
        W[:, 0] = eps * signs
        return W

    if out_dim < in_dim:
        rem = in_dim - out_dim
        W[:, out_dim:] = build_idinit_matrix(
            out_dim,
            rem,
            loosen_eps=0.0,
            alpha=-eps,
            device=device,
            dtype=dtype,
        )
    else:
        rows = torch.arange(out_dim, device=device)
        neg_cols = (rows + 1) % in_dim
        W[rows, neg_cols] = -eps

    return W


@torch.no_grad()
def idinit_residual_last_(
    layer: nn.Module,
    *,
    eps: float = 1e-6,
    bias_zero: bool = True,
) -> nn.Module:
    """
    Paper-faithful residual-last initialization:
      - Linear   -> IDIZ_eps
      - Conv2d   -> IDIZC_eps via patch-maintain reshape
    """
    if isinstance(layer, nn.Linear):
        W = build_idiz_matrix(
            layer.out_features,
            layer.in_features,
            eps=eps,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        layer.weight.copy_(W)

        if layer.bias is not None and bias_zero:
            layer.bias.zero_()
        return layer

    if isinstance(layer, nn.Conv2d):
        out_c, in_c, kH, kW = layer.weight.shape

        flat_in = in_c * kH * kW
        W2 = build_idiz_matrix(
            out_c,
            flat_in,
            eps=eps,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )

        # Keep same reshape convention as your current IDIC code
        # so we only change IDIZ behavior, not tensor ordering.
        W4 = W2.view(out_c, in_c, kH, kW)

        layer.weight.copy_(W4)

        if layer.bias is not None and bias_zero:
            layer.bias.zero_()
        return layer

    raise TypeError(f"Unsupported layer type: {type(layer)}")




