"""Plain PyTorch copy of mbconv_vae.py for standalone Inductor capture.

The architecture, initialization, parameter names and stochastic sampling match
MBConvVAE_V2. BlurPool uses its native Torch recipe without DeepOpt registration.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def blur_pool(x, kernel, stride, padding, channels):
    padded = F.pad(x, padding, mode="reflect")
    return F.conv2d(padded, kernel.to(x.dtype), stride=stride, groups=channels)


def norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class BlurPool(nn.Module):
    def __init__(self, channels, filt_size=4, stride=2):
        super().__init__()
        if filt_size < 1:
            raise ValueError("filt_size must be >= 1")
        self.stride = stride
        coeff = torch.tensor(
            [math.comb(filt_size - 1, i) for i in range(filt_size)], dtype=torch.float32
        )
        kernel = torch.outer(coeff, coeff)
        kernel = kernel / kernel.sum()
        self.register_buffer(
            "kernel", kernel.view(1, 1, filt_size, filt_size).repeat(channels, 1, 1, 1)
        )
        left = (filt_size - 1) // 2
        right = (filt_size - 1) - left
        self.padding = (left, right, left, right)
        self.channels = channels

    def forward(self, x):
        return blur_pool(x, self.kernel, self.stride, self.padding, self.channels)


class SEBlock(nn.Module):
    def __init__(self, channels, sequeeze_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, sequeeze_channels), 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(max(1, sequeeze_channels), channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class SwiGLU(nn.Module):
    def __init__(self, channels, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = (8 * channels) // 3
        self.gate_proj = nn.Conv2d(channels, hidden, 1, bias=False)
        self.up_proj = nn.Conv2d(channels, hidden, 1, bias=False)
        self.down_proj = nn.Conv2d(hidden, channels // 2, 1, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MBConv(nn.Module):
    def __init__(self, c_in, c_out, expansion=4, stride=1):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden = c_in * expansion
        mlp_hidden = c_out * 2
        self.body = nn.Sequential(
            norm(c_in),
            nn.Conv2d(c_in, hidden, 1, bias=False),
            norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            SEBlock(hidden, c_in // 4),
            norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, c_out, 1, bias=False),
            norm(c_out),
        )
        self.mlp_norm = norm(c_out * 2)
        self.mlp = SwiGLU(c_out * 2, mlp_hidden)
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.skip = (
            nn.Identity() if c_in == c_out and stride == 1 else nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, bias=False),
                BlurPool(c_out, filt_size=4, stride=stride) if stride != 1 else nn.Identity(),
                norm(c_out),
            )
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.body(x)
        x = torch.cat([skip, x], dim=1)
        return skip + self.res_scale * self.mlp(self.mlp_norm(x))


class MBConvVAE_V2(nn.Module):
    def __init__(self, base_channels=32, blocks_per_stage=2, latent_channels=8, max_channels=512):
        super().__init__()
        if min(base_channels, max_channels, blocks_per_stage, latent_channels) < 1:
            raise ValueError("All architecture parameters must be positive integers.")
        if max_channels < base_channels:
            raise ValueError("max_channels must be >= base_channels")

        self.latent_channel = latent_channels
        encoder = [nn.Conv2d(3, base_channels, 3, padding=1, bias=False)]
        c_in = base_channels
        stages = []
        growth = (max_channels - base_channels) // (3 * blocks_per_stage)
        for _ in range(3):
            transitions = []
            for _ in range(blocks_per_stage):
                c_out = c_in + growth
                encoder.append(MBConv(c_in, c_out))
                transitions.append((c_in, c_out))
                c_in = c_out
            encoder.append(MBConv(c_in, c_in, stride=1))
            encoder.append(BlurPool(c_in, filt_size=4, stride=2))
            stages.append(transitions)
        encoder += [norm(c_in), nn.SiLU()]
        self.encoder = nn.Sequential(*encoder)
        self.moments = nn.Conv2d(c_in, 2 * latent_channels, 1, bias=False)

        self.from_latent = nn.Conv2d(latent_channels, c_in, 3, padding=1, bias=False)
        decoder = []
        for transitions in reversed(stages):
            decoder.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False),
            ])
            for prev, curr in reversed(transitions):
                decoder.append(MBConv(curr, prev))
                c_in = prev
        decoder += [
            norm(c_in), nn.SiLU(), nn.Conv2d(c_in, 3, 3, padding=1, bias=False), nn.Tanh()
        ]
        self.decoder = nn.Sequential(*decoder)
        self.apply(self.init_weights)

    @torch.no_grad()
    def init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            if module.groups == module.in_channels and module.in_channels > 1:
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            else:
                nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm) and module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        mu, logvar = self.moments(self.encoder(x)).chunk(2, dim=1)
        return mu, logvar.clamp(-20.0, 10.0)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z, output_size=None):
        x = self.decoder(self.from_latent(z))
        if output_size is not None:
            h, w = output_size
            x = x[..., :h, :w]
        return x

    def forward(self, x, sample=None):
        mu, logvar = self.encode(x)
        # Preserve the supplied model: validation also samples z; sample is unused.
        z = self.reparameterize(mu, logvar)
        h, w = x.shape[-2:]
        return self.decode(z, (h, w)), mu, logvar
