import torch
from torch import nn
import torch.nn.functional as F

import Deep_Optimization.Activation_Compression.modules.layers as layers

import math


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def _fan_in(conv: nn.Conv2d):
    # return conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    c_out, c_in, kH, kW = conv.weight.shape
    return (conv.in_channels // conv.groups) * kH * kW


def _target_std(conv, nonlinearity="relu"):
    gain = nn.init.calculate_gain(nonlinearity)
    return gain / math.sqrt(_fan_in(conv))


def _normalize_per_out_channel(w, target_std, eps=1e-6):
    std = w.std(dim=(1,2,3), keepdim=True, unbiased=False).clamp_min(eps)
    return w * (target_std / std)


def _radial_grid(k, device):
    fy = torch.fft.fftfreq(k, device=device)
    fx = torch.fft.fftfreq(k, device=device)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")

    r = torch.sqrt(fy**2 + fx**2)
    r = r / r.max().clamp_min(1e-12)
    return r


# smooth interpolation instead of hard bins
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


def _interp_radial_prior_phase(radial_prior_phase, k, device):
    radial_prior_phase = radial_prior_phase.to(device)
    num_bins = radial_prior_phase.numel()
    r = _radial_grid(k, device)
    pos = r * (num_bins - 1)
    lo = pos.floor().long()
    hi = torch.clamp(lo + 1, max=num_bins - 1)
    w = pos - lo

    # convert phase to circular representation
    cos_prior = torch.cos(radial_prior_phase)
    sin_prior = torch.sin(radial_prior_phase)
    cos_interp = cos_prior[lo] * (1 - w) + cos_prior[hi] * w
    sin_interp = sin_prior[lo] * (1 - w) + sin_prior[hi] * w
    phase = torch.atan2(sin_interp, cos_interp)
    return phase


# whitening
def whiten_factor(r, gamma=0.5):
    return (r + 1e-3).pow(gamma).clamp(max=1.5)


# -------------------------------------------------
# Spectral Shaping Initialization
# -------------------------------------------------

def spectral_shape_init(
    conv,
    radial_prior_power,
    alpha=0.05,
    gamma=1.2,
    phase_noise=0.05,
    nonlinearity="relu",
):

    out_c, in_c, k, k2 = conv.weight.shape

    if k != k2 or k == 1:
        nn.init.orthogonal_(conv.weight.view(out_c, -1))
        return

    device = conv.weight.device

    # 1 orthogonal base
    w = torch.empty(out_c, in_c, k, k, device=device)
    nn.init.orthogonal_(w.view(out_c, -1))
    w += 0.005 * torch.randn_like(w)

    # 2 FFT
    Wf = torch.fft.fft2(w, norm="ortho")

    amp = torch.abs(Wf).clamp_min(1e-12)
    phase = Wf / amp

    # 3 prior map
    prior_power = _interp_radial_prior(radial_prior_power, k, device)
    # prior_phase = _interp_radial_prior_phase(radial_prior_phase, k, device)

    r = _radial_grid(k, device)
    prior_power = prior_power * whiten_factor(r, gamma)
    prior_power = prior_power / prior_power.mean()

    # gain = (1 - alpha) + alpha * prior_power
    gain = torch.exp(alpha * torch.log(prior_power.clamp_min(1e-6)))
    

    # 4 phase noise for diversity
    # noise = torch.exp(1j * torch.randn_like(amp) * phase_noise)
    # shared_noise = torch.exp(1j * torch.randn(out_c,1,k,k, device=device) * phase_noise)

    noise = torch.randn(out_c, 1, k, k, device=device)
    noise = F.avg_pool2d(noise, kernel_size=3, stride=1, padding=1)
    shared_noise = torch.exp(1j * noise * phase_noise)
    phase = phase * shared_noise
    

    new_amp = amp * gain[None, None]
    Wf_new = phase * new_amp
    w_new = torch.fft.ifft2(Wf_new, norm="ortho").real

    # 5 preserve variance
    target_std = _target_std(conv, nonlinearity)
    w_new = _normalize_per_out_channel(w_new, target_std)

    conv.weight.data.copy_(w_new)

    if conv.bias is not None:
        conv.bias.data.zero_()


# -------------------------------------------------
# Model Initialization
# -------------------------------------------------

def frequency_aware_init(
    model,
    radial_prior_power,
    alpha=0.05,
    gamma=0.05,
    phase_noise=0.05,
    nonlinearity='relu'
):
    for n, m in model.named_modules():
        if isinstance(m, (layers.DOConv2d, nn.Conv2d)):
            spectral_shape_init(
                m,
                radial_prior_power,
                alpha=alpha,
                gamma=gamma,
                phase_noise=phase_noise,
                nonlinearity=nonlinearity
            )









