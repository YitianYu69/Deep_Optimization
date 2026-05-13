import torch
from torch import nn
import torch.nn.functional as F

import Deep_Optimization.Activation_Compression.modules.layers as layers

import math


def _fan_in(conv: nn.Conv2d):
    return conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]


def _target_std(conv, nonlinearity="relu"):
    gain = nn.init.calculate_gain(nonlinearity)
    return gain / math.sqrt(_fan_in(conv))


def _normalize_per_out_channel(w, target_std, eps=1e-6):
    std = w.std(dim=(1, 2, 3), keepdim=True, unbiased=False).clamp_min(eps)
    return w * (target_std / std)


def _radial_grid(k, device):
    fy = torch.fft.fftfreq(k, device=device)
    fx = torch.fft.fftfreq(k, device=device)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(fx**2 + fy**2)
    r = r / r.max().clamp_min(1e-12)
    return r


def _angle_grid(k, device):
    fy = torch.fft.fftfreq(k, device=device)
    fx = torch.fft.fftfreq(k, device=device)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    theta = torch.atan2(fy, fx)  # [-pi, pi]
    return theta


def _interp_radial_prior(radial_prior, k, device):
    radial_prior = radial_prior.to(device)
    num_bins = radial_prior.numel()
    r = _radial_grid(k, device)
    pos = r * (num_bins - 1)
    lo = pos.floor().long()
    hi = torch.clamp(lo + 1, max=num_bins - 1)
    w = pos - lo
    amp = radial_prior[lo] * (1 - w) + radial_prior[hi] * w
    return amp


def _soft_bandpass(r, center=0.4, width=0.15):
    return torch.exp(-0.5 * ((r - center) / max(width, 1e-6)) ** 2)

def _soft_lowpass(r, cutoff=0.25, sharpness=12.0):
    # smooth low-pass mask in [0,1]
    return torch.sigmoid((cutoff - r) * sharpness)

def _soft_highband(r, center=0.75, width=0.12):
    return torch.exp(-0.5 * ((r - center) / max(width, 1e-6)) ** 2)

def _orientation_mask(theta, angle_center, strength=1.0, power=2.0):
    """
    Orientation-aware mask in Fourier domain.
    Uses pi-periodicity because opposite directions correspond to same orientation.
    angle_center in radians.
    """
    # orientation equivalence under theta -> theta + pi
    delta = theta - angle_center
    # cos(2*delta) gives pi-periodic orientation similarity
    mask = ((torch.cos(2.0 * delta) + 1.0) * 0.5).clamp(0, 1)
    mask = mask ** power
    return 1.0 + strength * (mask - mask.mean())

def _bounded_gain_map(base_map, alpha=0.08, clip=0.25, mode='linear'):
    base_map = base_map / base_map.mean().clamp_min(1e-6)

    if mode == 'linear':
        delta = base_map - 1.0
        delta = delta.clamp(-clip, clip)
        return 1.0 + alpha * delta
    elif mode == 'log':
        log_map = torch.log(base_map.clamp_min(1e-6) + 1e-12)
        return torch.exp(alpha * log_map.clamp(-clip, clip))
    else:
        return 1.0
        

def _make_group_slices(out_c, ratios=(0.25, 0.25, 0.25, 0.25)):
    counts = [int(out_c * x) for x in ratios]
    counts[-1] = out_c - sum(counts[:-1])

    slices = []
    start = 0
    for c in counts:
        end = start + c
        slices.append(slice(start, end))
        start = end
    return slices


def _build_group_gain_maps(
    k,
    device,
    radial_prior_power,
    alpha=0.08,
    angular_beta=0.04,
    mid_center=0.42,
    mid_width=0.16,
):
    """
    Returns a dict of gain maps:
      low, mid, oriented_hv, baseline
    Each is [k, k], multiplicative, close to 1.
    """
    r = _radial_grid(k, device)
    theta = _angle_grid(k, device)

    # dataset radial envelope
    radial = _interp_radial_prior(radial_prior_power, k, device)
    radial = radial / radial.mean().clamp_min(1e-6)

    # 1) Low-frequency group:
    # keep teacher radial bias, but softly encourage low-pass stability
    low_map = radial * (0.75 + 0.25 * _soft_lowpass(r, cutoff=0.28, sharpness=12.0))
    low_gain = _bounded_gain_map(low_map, alpha=alpha, clip=0.20)

    # 2) Mid-band group:
    # modest bump in middle frequencies; do not boost extreme high freq
    mid_band = _soft_bandpass(r, center=mid_center, width=mid_width)
    mid_map = radial * (1.0 + 0.45 * (mid_band - mid_band.mean()))
    mid_gain = _bounded_gain_map(mid_map, alpha=alpha, clip=0.20)

    # 3) High-frequency group:
    high_band = _soft_highband(r, center=0.75, width=0.12)
    high_map = radial * (1.0 + 0.25 * (high_band - high_band.mean()))
    high_gain = _bounded_gain_map(high_map, alpha=alpha, clip=0.20)

    # 4) Oriented group:
    # use a safe radial envelope + weak angular specialization
    oriented_base = radial * (0.85 + 0.15 * _soft_bandpass(r, center=0.35, width=0.18))

    # horizontal-ish orientation (theta = 0) and vertical-ish (theta = pi/2)
    ori_h = _orientation_mask(theta, angle_center=0.0, strength=angular_beta, power=2.0)
    ori_v = _orientation_mask(theta, angle_center=math.pi / 2.0, strength=angular_beta, power=2.0)
    ori_d1 = _orientation_mask(theta, math.pi/4, strength=angular_beta)
    ori_d2 = _orientation_mask(theta, 3*math.pi/4, strength=angular_beta)

    oriented_h_gain = _bounded_gain_map(oriented_base * ori_h, alpha=alpha, clip=0.18)
    oriented_v_gain = _bounded_gain_map(oriented_base * ori_v, alpha=alpha, clip=0.18)
    oriented_d1_gain = _bounded_gain_map(oriented_base * ori_d1, alpha=alpha, clip=0.18)
    oriented_d2_gain = _bounded_gain_map(oriented_base * ori_d2, alpha=alpha, clip=0.18)

    # 5) Baseline group:
    # almost unchanged, just mild radial shaping
    baseline_gain = _bounded_gain_map(radial, alpha=alpha * 0.5, clip=0.12, mode='log')

    return {
        "low": low_gain,
        "mid": mid_gain,
        "high": high_gain,
        "oriented_h": oriented_h_gain,
        "oriented_v": oriented_v_gain,
        "oriented_d1": oriented_d1_gain,
        "oriented_d2": oriented_d2_gain,
        "baseline": baseline_gain,
    }

def _orthogonalize_rows(w, eps=1e-8):
    
    if w.numel() == 0 or w.shape[0] <= 1:
        return w

    out_c = w.shape[0]
    w_flat = w.view(out_c, -1)
    flat_dim = w_flat.shape[1]

    if out_c <= flat_dim:
        # normal case
        Q, _ = torch.linalg.qr(w_flat.T)
        w_orth = Q.T[:out_c]

    else:
        # more filters than dimensions
        Q, _ = torch.linalg.qr(w_flat)
        w_orth = Q[:, :flat_dim]

        # expand with small noise to fill rows
        if w_orth.shape[0] < out_c:
            extra = torch.randn(
                out_c - w_orth.shape[0],
                flat_dim,
                device=w.device
            ) * 0.01
            w_orth = torch.cat([w_orth, extra], dim=0)

    return w_orth.view_as(w)

def layer_adaptive_ratios(layer_idx, total_layers):

    t = layer_idx / max(total_layers - 1, 1)

    # early-layer targets
    high0 = 0.35
    mid0 = 0.25
    oriented0 = 0.25
    low0 = 0.05

    # late-layer targets
    high1 = 0.05
    mid1 = 0.25
    oriented1 = 0.35
    low1 = 0.25

    baseline = 0.10

    # smooth interpolation
    high = high0 * (1 - t) + high1 * t
    mid = mid0 * (1 - t) + mid1 * t
    oriented = oriented0 * (1 - t) + oriented1 * t
    low = low0 * (1 - t) + low1 * t

    ratios = [low, mid, oriented, baseline, high]

    # normalize
    s = sum(ratios)
    ratios = [r / s for r in ratios]

    return ratios

def spectral_shape_init_specialized(
    conv,
    radial_prior_power,
    alpha=0.08,
    phase_noise=0.03,
    nonlinearity="relu",
    layer_index=0,
    total_layers=80,
):
    """
    Filter-bank-specialized spectral init.
    Works best on the first 1-2 spatial conv layers.
    """
    out_c, in_c, k, k2 = conv.weight.shape

    if k != k2:
        nn.init.orthogonal_(conv.weight.view(out_c, -1))
        if conv.bias is not None:
            conv.bias.data.zero_()
        return

    device = conv.weight.device

    # orthogonal base + tiny noise
    w = torch.empty(out_c, in_c, k, k, device=device)
    nn.init.orthogonal_(w.view(out_c, -1))
    w += 0.02 * torch.randn_like(w)

    Wf = torch.fft.fft2(w, norm="ortho")
    amp = torch.abs(Wf).clamp_min(1e-12)
    phase = Wf / amp

    # mild global phase noise to avoid brittle structure
    shared_noise = torch.exp(1j * torch.randn(out_c, 1, k, k, device=device) * phase_noise)
    phase = phase * shared_noise

    gain_maps = _build_group_gain_maps(
        k=k,
        device=device,
        radial_prior_power=radial_prior_power,
        alpha=alpha,
        angular_beta=0.05,
        mid_center=0.42,
        mid_width=0.16,
    )

    # Split output filters into 4 groups:
    # low / mid / oriented / baseline
    # Inside oriented, split half to H and half to V
    ratios = layer_adaptive_ratios(layer_index, total_layers)
    low_sl, mid_sl, ori_sl, base_sl, high_sl = _make_group_slices(out_c, ratios=ratios)
    # low_sl, mid_sl, ori_sl, base_sl, high_sl = _make_group_slices(out_c, ratios=(0.225, 0.225, 0.225, 0.225, 0.10))
    # low_sl, mid_sl, ori_sl, base_sl, high_sl = _make_group_slices(out_c, ratios=(0.25, 0.25, 0.25, 0.25, 0.25))

    new_amp = amp.clone()

    if low_sl.stop > low_sl.start:
        new_amp[low_sl] *= gain_maps["low"][None, None]

    if mid_sl.stop > mid_sl.start:
        new_amp[mid_sl] *= gain_maps["mid"][None, None]

    if high_sl.stop > high_sl.start:
        new_amp[high_sl] *= gain_maps["high"][None, None]

    if ori_sl.stop > ori_sl.start:
        ori_count = ori_sl.stop - ori_sl.start
        split = ori_sl.start + ori_count // 4

        if split > ori_sl.start:
            new_amp[ori_sl.start:split] *= gain_maps["oriented_h"][None, None]
        if 2 * split > split:
            new_amp[split:2 * split] *= gain_maps["oriented_d1"][None, None]
        if 3 * split > 2 * split:
            new_amp[2 * split:3 * split] *= gain_maps["oriented_v"][None, None]
        if ori_sl.stop > 3 * split:
            new_amp[3 * split:ori_sl.stop] *= gain_maps["oriented_d2"][None, None]

    if base_sl.stop > base_sl.start:
        new_amp[base_sl] *= gain_maps["baseline"][None, None]

    Wf_new = phase * new_amp
    w_new = torch.fft.ifft2(Wf_new, norm="ortho").real

    # normalize each output filter
    w_flat = w_new.view(out_c, -1)
    w_flat = F.normalize(w_flat, dim=1)
    w_new = w_flat.view(out_c, in_c, k, k)

    target_std = _target_std(conv, nonlinearity)
    w_new = _normalize_per_out_channel(w_new, target_std)


    if low_sl.stop > low_sl.start:
        w_new[low_sl] = _orthogonalize_rows(w_new[low_sl])

    if mid_sl.stop > mid_sl.start:
        w_new[mid_sl] = _orthogonalize_rows(w_new[mid_sl])

    if high_sl.stop > high_sl.start:
        w_new[high_sl] = _orthogonalize_rows(w_new[high_sl])

    if ori_sl.stop > ori_sl.start:
        ori_count = ori_sl.stop - ori_sl.start
        split = ori_sl.start + ori_count // 4

        if split > ori_sl.start:
            w_new[ori_sl.start:split] = _orthogonalize_rows(w_new[ori_sl.start:split])
        if 2 * split > split:
            w_new[split:2 * split] = _orthogonalize_rows(w_new[split:2 * split])
        if 3 * split > 2 * split:
            w_new[2 * split:3 * split] = _orthogonalize_rows(w_new[2 * split:3 * split])
        if ori_sl.stop > 3 * split:
            w_new[3 * split:ori_sl.stop] = _orthogonalize_rows(w_new[3 * split:ori_sl.stop])

    if base_sl.stop > base_sl.start:
        w_new[base_sl] = _orthogonalize_rows(w_new[base_sl])

    conv.weight.data.copy_(w_new)

    if conv.bias is not None:
        conv.bias.data.zero_()


def frequency_aware_init_specialized(
    model,
    radial_prior_power,
    alpha=0.08,
    phase_noise=0.03
):
    conv_idx = 0
    convs = [m for m in model.modules() if isinstance(m, (nn.Conv2d, layers.DOConv2d))]

    for _, m in model.named_modules():
        if isinstance(m, (layers.DOConv2d, nn.Conv2d)):
            conv = convs[conv_idx]

            spectral_shape_init_specialized(
                conv,
                radial_prior_power=radial_prior_power,
                alpha=alpha,
                phase_noise=phase_noise,
                nonlinearity="relu",
                layer_index=conv_idx,
                total_layers=len(convs),
            )
            # else:
            #     nn.init.orthogonal_(conv.weight.view(conv.weight.shape[0], -1))
            #     if conv.bias is not None:
            #         conv.bias.data.zero_()

            conv_idx += 1



