import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import v2

from Deep_Optimization.Initialization.freq_init_specialized import spectral_shape_init_specialized
from Deep_Optimization.Initialization.freq_init import spectral_shape_init
from Deep_Optimization.Initialization.Fully_Identity_init import idinit_conv2d_patch_, idinit_linear_
from Deep_Optimization.Initialization.ZerO_init import init_ZerO_convolution, init_ZerO_linear

import Deep_Optimization.Activation_Compression.modules.layers as layers

from Deep_Optimization.Adversarial_Attack.FGSM import PGD_attack, FGSM_attack


def _radial_grid(h, w, device):
    fy = torch.fft.fftfreq(h, d=1.0).to(device)
    fx = torch.fft.fftfreq(w, d=1.0).to(device)
    gy, gx = torch.meshgrid(fy, fx, indexing='ij')
    r = torch.sqrt(gx**2 + gy**2)
    r = r / r.max()
    return r


def _radial_bin(power_map, num_bins=64):
    """
    power_map: [H, W]
    return: [num_bins]
    """
    H, W = power_map.shape
    device = power_map.device

    r = _radial_grid(H, W, device)

    bins = torch.linspace(0, 1, num_bins + 1, device=device)
    radial_power = torch.zeros(num_bins, device=device)

    for i in range(num_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.any():
            radial_power[i] = power_map[mask].mean()

    return radial_power


@torch.no_grad()
def compute_radial_prior_power(
    dataloader,
    device,
    num_batches=100000,
    num_bins=64,
    attack=False,
    model=None,
    cri={},
    eps=1e-8,
    mu=torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1),
    std=torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1),
):
    """
    Compute dataset radial power spectrum

    Returns:
        radial_prior_power: [num_bins]
    """

    radial_accum = torch.zeros(num_bins, device=device)
    count = 0

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        images = batch[0].to(device)   # [B, C, H, W]
        if attack:
            images, labels = batch
            images, label = images.to(device), labels.to(device)

            pgd_images = PGD_attack(model, cri['Valid'], images, label, num_iters=7, random_eps=8/255, alpha=10/255)
            
            images = images * std + mu
            images = torch.cat([images, pgd_images], dim=0)
            images = v2.Normalize(mu, std)(images)


        images = images - images.mean(dim=(-2, -1), keepdim=True)

        Xf = torch.fft.fft2(images, dim=(-2, -1), norm="ortho")

        # power spectrum
        power = (Xf.real**2 + Xf.imag**2)  # [B, C, H, W]

        # average over batch + channels
        power = power.mean(dim=(0, 1))  # [H, W]
        radial = _radial_bin(power, num_bins=num_bins)

        radial_accum += radial
        count += 1

    radial_prior = radial_accum / max(count, 1)

    radial_prior = radial_prior / (radial_prior.mean() + eps)

    return radial_prior


def compute_num_layers(model):
    layers = 0
    for _ in model.modules():
        layers += 1
    return layers


def init(
        model: nn.Module, 
        freq_init: bool = True,
        num_bins: int = 32,
        dataloader: DataLoader = None,
        num_batches: int = 100,
        alpha: float = 0.05,
        gamma: float = 0.05,
        phase_noise: float =0.05,

        orthogonal: bool = True,
        zero_init: bool = False,
        ID_init: bool = False,

        attack: bool = False,
        cri: dict = {},
        mu: torch.Tensor = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1),
        std: torch.Tensor = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1),

        device: str = 'cuda'
):
    if freq_init:
        radial_prior_power = compute_radial_prior_power(dataloader, device=device, num_batches=num_batches, num_bins=num_bins, attack=attack, model=model, cri=cri, mu=mu, std=std)  
        num_layers = compute_num_layers(model)

    for i, m in enumerate(model.modules()):
        if isinstance(m, (layers.DOConv2d, nn.Conv2d)):
            is_full = (m.groups == 1) and (m.kernel_size[0] == m.kernel_size[1]) and m.kernel_size[0] > 1
            is_depth  = (m.groups == m.in_channels == m.out_channels) and (m.kernel_size[0] == m.kernel_size[1]) and m.kernel_size[0] > 1
            is_group = (m.groups > 1) and not (m.groups == m.in_channels == m.out_channels) and m.kernel_size[0] == m.kernel_size[1] and m.kernel_size[0] > 1

            if freq_init and is_full:
                spectral_shape_init_specialized(
                    m,
                    radial_prior_power,
                    alpha=alpha,
                    phase_noise=phase_noise,
                    layer_index=i,
                    total_layers=num_layers,
                )

            elif freq_init and is_depth:
                spectral_shape_init(
                    m,
                    radial_prior_power,
                    alpha=alpha,
                    gamma=gamma,
                    phase_noise=phase_noise
                )
            elif freq_init and is_group:
                spectral_shape_init(
                    m,
                    radial_prior_power,
                    alpha=alpha,
                    gamma=gamma,
                    phase_noise=phase_noise
                )
            else:
                if orthogonal:
                    
                    out_c, in_c, k1, k2 = m.weight.shape
                    new_w = m.weight.view(out_c, -1)
                    nn.init.orthogonal_(new_w)
                elif zero_init:
                    init_ZerO_convolution(m.weight)
                elif ID_init:
                    idinit_conv2d_patch_(m)

        elif isinstance(m, (layers.DOLinear, nn.Linear)):
            # if orthogonal:
            #     nn.init.orthogonal_(m.weight)
            # elif zero_init:
            # init_ZerO_linear(m.weight)
            # elif ID_init:
            idinit_linear_(m)  
        
