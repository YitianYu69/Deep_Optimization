import math
import torch
from torch import nn
import torch.nn.functional as F

def radial_spectrum_2d(
    x,
    raw=False,
    num_rad_bins=16,
    mode='magnitude',
    eps=1e-8,
    out_dtype=None,
    *,
    vit_grid=None,              # (H, W) explicit patch grid
    num_prefix_tokens=None,      # number of prefix tokens total
    max_prefix_tokens=16
):
    """
    Supports:
      - CNN feature maps: x [B, C, H, W]
      - ViT tokens:       x [B, N, D]  (tokens, embed dim)

    Returns:
      spectrum [num_bins], normalized to sum=1 (aggregated across batch).
    """

    if out_dtype is None:
        out_dtype = x.dtype

    if x.dim() == 3:
        # ViT tokens: [B, N, D]
        B, N, D = x.shape

        # Determine grid size and how many prefix tokens exist.
        if vit_grid is not None:
            H, W = vit_grid
            needed = H * W
            if num_prefix_tokens is None:
                num_prefix_tokens = N - needed
            if N - num_prefix_tokens != needed:
                raise ValueError(
                    f"vit_grid={vit_grid} implies {needed} patch tokens, but got N={N} "
                    f"with num_prefix_tokens={num_prefix_tokens} => {N - num_prefix_tokens} tokens."
                )
        else:
            # Auto-infer num_prefix_tokens by finding N-k that is a perfect square
            if num_prefix_tokens is None:
                found = None
                for k in range(0, max_prefix_tokens + 1):
                    m = N - k
                    if m <= 0:
                        break
                    s = int(math.isqrt(m))
                    if s * s == m:
                        found = k
                        H = W = s
                        break
                if found is None:
                    raise ValueError(
                        f"Cannot infer square token grid from N={N}. "
                        f"Pass vit_grid=(H,W) or num_prefix_tokens explicitly."
                    )
                num_prefix_tokens = found
            else:
                m = N - num_prefix_tokens
                s = int(math.isqrt(m))
                if s * s != m:
                    raise ValueError(
                        f"N - num_prefix_tokens = {m} is not a perfect square. "
                        f"Pass vit_grid=(H,W) instead."
                    )
                H = W = s

        # Default: patch-token spectrum (original behavior)
        if num_prefix_tokens > 0:
            x = x[:, num_prefix_tokens:, :].contiguous()  # [B, H*W, D]
        else:
            x = x.contiguous()

        # [B, H*W, D] -> [B, H, W, D] -> [B, D, H, W]
        x = x.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

    elif x.dim() != 4:
        raise ValueError(f"Expected x dim 3 or 4, got {x.dim()}")
    
    

    # CNN path (or converted ViT path): x [B, C, H, W]
    # x = x.mean(dim=1)                 # [B, H, W]
    x = x.pow(2).mean(dim=1).sqrt()   # RMS over channels [B, H, W]

    # FFT in fp32 for safety/stability
    x_fft = x.to(torch.float32)
    F = torch.fft.fft2(x_fft)         # complex64
    F = torch.fft.fftshift(F)  # shift only spatial dims

    B, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device='cuda'),
        torch.arange(W, device='cuda'),
        indexing="ij"
    )
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    dy = yy - cy
    dx = xx - cx

    def produce_freq_magnitude(raw=False):
        P = (F.real * F.real + F.imag * F.imag)  # [B, H, W] float32

        if not raw:
            # ----- Radius -----
            r = torch.sqrt((dy) ** 2 + (dx) ** 2)
            r = (r / (r.max() + 1e-12) * (num_rad_bins - 1)).to(torch.long)  # [H, W]

            # ----- Polar bin index -----
            bin_1d = r
            num_total_bins = num_rad_bins

            spectrum = torch.zeros(num_total_bins, device=P.device, dtype=torch.float32)
            idx = bin_1d.flatten().expand(B, -1)     # [B, HW]
            vals = P.reshape(B, -1)            # [B, HW]
            spectrum.scatter_add_(0, idx.reshape(-1), vals.reshape(-1))

            spectrum = spectrum / (spectrum.sum() + eps)
            return spectrum.view(num_rad_bins)
        else:
            return P

    def produce_freq_phase(raw=False):
        phase = torch.angle(F)  # [B, H, W]

        if not raw:
            phasor_cos = torch.cos(phase)
            phasor_sin = torch.sin(phase)

            spec_real = torch.zeros(num_rad_bins, device='cuda')
            spec_imag = torch.zeros(num_rad_bins, device='cuda')

            # ----- Radius -----
            r = torch.sqrt((dy) ** 2 + (dx) ** 2)
            r = (r / (r.max() + 1e-12) * (num_rad_bins - 1)).to(torch.long)  # [H, W]

            idx = r.flatten().expand(B, -1)  # [B, HW]
            mag = torch.abs(F)
            weight = mag ** 2
            vals_r = (weight * phasor_cos).reshape(B, -1)
            vals_i = (weight * phasor_sin).reshape(B, -1)

            spec_real.scatter_add_(0, idx.reshape(-1), vals_r.reshape(-1))
            spec_imag.scatter_add_(0, idx.reshape(-1), vals_i.reshape(-1))

            coherence = torch.sqrt(spec_real**2 + spec_imag**2) / (weight.sum() + eps)
            return coherence
        else:
            mag = torch.abs(F)
            phase = torch.where(mag > 1e-6, phase, torch.zeros_like(phase))
            return phase

    if mode == 'magnitude':
        return produce_freq_magnitude(raw)
    
    elif mode == 'phase':
        return produce_freq_phase(raw)
    else:
        return produce_freq_magnitude(raw) + produce_freq_phase(raw)






# import torch
# import math
# import torch.nn.functional as F


# # ---------------------------------------------------
# # DCT-II (orthonormal), separable 2D implementation
# # ---------------------------------------------------

# def dct_1d(x, norm="ortho"):
#     """
#     x: [..., N]
#     returns DCT-II along last dim
#     """
#     N = x.shape[-1]

#     # Even-symmetric extension
#     x_ext = torch.cat([x, x.flip(-1)], dim=-1)

#     # FFT
#     X = torch.fft.fft(x_ext, dim=-1)

#     # Take real cosine part
#     k = torch.arange(N, device=x.device)
#     W = torch.exp(-1j * math.pi * k / (2 * N))

#     out = (X[..., :N] * W).real * 2

#     if norm == "ortho":
#         out[..., 0] /= math.sqrt(4 * N)
#         out[..., 1:] /= math.sqrt(2 * N)
#     return out


# def dct_2d(x):
#     """
#     x: [B,H,W]
#     """
#     x = dct_1d(x, norm="ortho")
#     x = dct_1d(x.transpose(-1, -2), norm="ortho").transpose(-1, -2)
#     return x


# # ---------------------------------------------------
# # Radial DCT spectrum (your main function)
# # ---------------------------------------------------

# def radial_spectrum_2d(
#     x,
#     num_bins=16,
#     eps=1e-8,
#     out_dtype=None,

#     *,
#     vit_grid=None,
#     num_prefix_tokens=None,
#     max_prefix_tokens=16,
#     use_cls_token=True,
# ):
#     """
#     Supports:
#       - CNN feature maps: x [B,C,H,W]
#       - ViT tokens:       x [B,N,D]

#     Returns:
#       spectrum [num_bins], normalized to sum=1
#     """

#     if out_dtype is None:
#         out_dtype = x.dtype

#     # ---------------- ViT handling ----------------

#     if x.dim() == 3:
#         B, N, D = x.shape

#         if vit_grid is not None:
#             H, W = vit_grid
#             needed = H * W
#             if num_prefix_tokens is None:
#                 num_prefix_tokens = N - needed
#         else:
#             if num_prefix_tokens is None:
#                 for k in range(max_prefix_tokens + 1):
#                     m = N - k
#                     s = int(math.isqrt(m))
#                     if s * s == m:
#                         num_prefix_tokens = k
#                         H = W = s
#                         break
#             else:
#                 m = N - num_prefix_tokens
#                 s = int(math.isqrt(m))
#                 H = W = s

#         if use_cls_token:
#             cls = x[:, 0, :]  # [B,D]
#             HW = H * W
#             if D < HW:
#                 cls = F.pad(cls, (0, HW - D))
#             elif D > HW:
#                 cls = cls[:, :HW]

#             x = cls.view(B, H, W).unsqueeze(1)

#         else:
#             if num_prefix_tokens > 0:
#                 x = x[:, num_prefix_tokens:, :]
#             x = x.view(B, H, W, D).permute(0, 3, 1, 2)

#     elif x.dim() != 4:
#         raise ValueError(f"Expected x dim 3 or 4, got {x.dim()}")

#     # ------------------------------------------------
#     # CNN path: x [B,C,H,W] → spatial mean
#     # ------------------------------------------------

#     x = x.mean(dim=1)   # [B,H,W]

#     # ------------------------------------------------
#     # DCT (fp32 for stability)
#     # ------------------------------------------------

#     x = x.float()
#     F = dct_2d(x)
#     P = F.pow(2)        # power spectrum [B,H,W]

#     B, H, W = P.shape

#     # ------------------------------------------------
#     # Radial bins
#     # ------------------------------------------------

#     yy, xx = torch.meshgrid(
#         torch.arange(H, device=P.device),
#         torch.arange(W, device=P.device),
#         indexing="ij"
#     )

#     r = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
#     r = r / (r.max() + 1e-12)
#     r = (r * (num_bins - 1)).long()

#     spectrum = torch.zeros(num_bins, device=P.device)

#     idx = r.flatten().expand(B, -1)
#     vals = P.reshape(B, -1)

#     spectrum.scatter_add_(0, idx.reshape(-1), vals.reshape(-1))

#     spectrum = spectrum / (spectrum.sum() + eps)

#     return spectrum.to(out_dtype)


@torch.compile(fullgraph=True)
def wasserstein(p, q, eps=1e-8):
    """
    p, q:
        1D → [K]
        2D → [R, A]
    """

    p32 = p.to(torch.float32)
    q32 = q.to(torch.float32)

    # Normalize
    p32 = p32 / (p32.sum() + eps)
    q32 = q32 / (q32.sum() + eps)

    if p32.ndim == 1:
        # ----- 1D Wasserstein -----
        cdf_p = torch.cumsum(p32, dim=0)
        cdf_q = torch.cumsum(q32, dim=0)

        return torch.sum(torch.abs(cdf_p - cdf_q))

    elif p32.ndim == 2:
        # ----- 2D Separable Wasserstein -----

        # Radial marginal
        p_r = p32.sum(dim=1)  # [R]
        q_r = q32.sum(dim=1)

        cdf_pr = torch.cumsum(p_r, dim=0)
        cdf_qr = torch.cumsum(q_r, dim=0)

        w_r = torch.sum(torch.abs(cdf_pr - cdf_qr))

        # Angular marginal
        p_a = p32.sum(dim=0)  # [A]
        q_a = q32.sum(dim=0)

        cdf_pa = torch.cumsum(p_a, dim=0)
        cdf_qa = torch.cumsum(q_a, dim=0)

        w_a = torch.sum(torch.abs(cdf_pa - cdf_qa))

        return w_r + w_a

    else:
        raise ValueError(f"Unsupported ndim: {p32.ndim}")
    

def log_huber_loss(model_spectrum, target_spectrum, delta=0.1):
    model_spectrum = model_spectrum.to(torch.float32)
    target_spectrum = target_spectrum.to(torch.float32)

    model_spectrum_log = torch.log1p(model_spectrum)
    target_spectrum_log = torch.log1p(target_spectrum)

    return F.huber_loss(model_spectrum_log, target_spectrum_log, delta=delta, reduction='mean')


@torch.compile(fullgraph=True)
def kl_div_spectrum(q, p, eps=1e-8):
    # q, p: [B,K], each row sums to 1
    q = q.clamp_min(eps)
    p = p.clamp_min(eps)
    return (q * (q.log() - p.log())).sum(dim=-1).mean()


@torch.compile(fullgraph=True)
def temp_kl(p, q, T=2.0, eps=1e-8):

    p = p.clamp_min(eps)
    q = q.clamp_min(eps)

    # Power tempering (equivalent to softmax(log p / T))
    pT = p ** (1.0 / T)
    qT = q ** (1.0 / T)

    pT = pT / pT.sum(dim=-1, keepdim=True)
    qT = qT / qT.sum(dim=-1, keepdim=True)

    log_pT = torch.log(pT)
    log_qT = torch.log(qT)

    kl = (pT * (log_pT - log_qT)).sum(dim=-1).mean()

    return (T * T) * kl


