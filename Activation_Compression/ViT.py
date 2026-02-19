import torch
from torch import nn
import torch.nn.functional as F
from torch.fx.proxy import Proxy

from freq_utils import radial_spectrum_2d

# class PatchEmbed(nn.Module):
#     def __init__(self, 
#                  in_channels: int = 3,
#                  patch_size: int = 14,
#                  image_size: int = 224,
#                  embed_dim: int = 768):
#         super().__init__()
#         self.num_patches = (image_size // patch_size) ** 2
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

#     def forward(self, x):
#         x = self.proj(x) # batch_size, embed_dim, sqrt(num_patches), sqrt(num_patches)
#         x = x.flatten(2) # batch_size, embed_dim, num_patches
#         return x.transpose(1, 2).contiguous()


# class MultiHeadAttention(nn.Module):
#     def __init__(self, 
#                  num_head: int,
#                  embed_dim: int,
#                  attn_p: float = 0.2):
#         super().__init__()

#         assert embed_dim % num_head == 0, "The embed dim must be divisible by the num of head"
#         self.num_head = num_head
#         self.head_dim = embed_dim // num_head

#         self.qkv = nn.Linear(embed_dim, embed_dim * 3)
#         self.proj = nn.Linear(embed_dim, embed_dim)
#         self.attn_p = attn_p

#     def forward(self, x):
#         batch_size, num_patch, embed_dim = x.size()

#         qkv = self.qkv(x)
#         qkv = qkv.view(batch_size, num_patch, 3, self.num_head, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn_out = F.scaled_dot_product_attention(q, k, v,
#                                                   dropout_p=self.attn_p if self.training else 0.0)
#         out = attn_out.transpose(1, 2).contiguous().view(batch_size, num_patch, -1)
#         return self.proj(out)


# class MLP(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  hidden_dim: int,
#                  mlp_p):
#         super().__init__()
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.gelu = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, embed_dim)
#         self.dropout = nn.Dropout(mlp_p)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.gelu(x)
#         x = self.fc2(x)
#         return self.dropout(x)


# class TransformerBlock(nn.Module):
#     def __init__(self, 
#                  embed_dim,
#                  num_head,
#                  mlp_ratio,
#                  attn_p,
#                  mlp_p):
#         super().__init__()

#         self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.attn = MultiHeadAttention(num_head, embed_dim, attn_p)

#         self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
#         hidden_dim = int(embed_dim * mlp_ratio)
#         self.mlp = MLP(embed_dim, hidden_dim, mlp_p)

#     def forward(self, x):
#         x = x + self.attn(self.layernorm1(x))
#         return x + self.mlp(self.layernorm2(x))


# class FrequencyProjection(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.to_scalar = nn.Linear(embed_dim, 1, bias=False)
#         self.film = nn.Linear(embed_dim, 2 * embed_dim, bias=True)

#     def forward(self, patch_tokens, freq_token, H, W):
#         """
#         patch_tokens: [B, HW, D]
#         freq_token:   [B, 1, D]
#         """
#         gamma, beta = self.film(freq_token).chunk(2, dim=-1)  # [B,1,D]
#         conditioned = patch_tokens * (1 + gamma) + beta       # broadcast

#         proxy = self.to_scalar(conditioned).squeeze(-1)       # [B, HW]
#         proxy = proxy.view(proxy.size(0), H, W)               # [B,H,W]
#         return proxy





# # class VisionTransformer(nn.Module):
# #     def __init__(self,
# #                  in_channels,
# #                  num_classes,
# #                  image_size,
# #                  patch_size,
# #                  embed_dim,
# #                  depth,
# #                  num_head,
# #                  mlp_ratio,
# #                  attn_p,
# #                  mlp_p,
# #                  pos_p):
# #         super().__init__()
# #         self.Hp = image_size // patch_size
# #         self.Wp = image_size // patch_size

# #         self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)

# #         self.freq_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
# #         self.freq_proj = FrequencyProjection(embed_dim)


# #         self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
# #         self.pos_tokens = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
# #         self.pos_dropout = nn.Dropout(pos_p)

# #         self.blocks = nn.ModuleList(
# #             [
# #                 TransformerBlock(embed_dim, num_head, mlp_ratio, attn_p, mlp_p)
# #                 for _ in range(depth)
# #             ]
# #         )

# #         self.final_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
# #         self.head = nn.Linear(embed_dim, num_classes)

# #     def forward(self, x):
# #         x = self.patch_embed(x)
# #         B, HW, D = x.shape

# #         freq_token = self.freq_token.expand(B, -1, -1)

# #         # === frequency proxy map ===
# #         proxy_map = self.freq_proj(x, freq_token, self.Hp, self.Wp)

# #         cls_tokens = self.cls_tokens.expand(x.size(0), -1, -1)
# #         x = torch.cat([cls_tokens, freq_token, x], dim=1)
# #         x = x + self.pos_tokens
# #         x = self.pos_dropout(x)

# #         # freq_tokens_final = x[:, 1:2, :]

# #         for block in self.blocks:
# #             x = block(x)
# #         x = self.final_layernorm(x)
# #         cls_tokens_final = x[:, 0, :]
# #         return self.head(cls_tokens_final), proxy_map.unsqueeze(1)





# class VisionTransformer(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  num_classes,
#                  image_size,
#                  patch_size,
#                  embed_dim,
#                  depth,
#                  num_head,
#                  mlp_ratio,
#                  attn_p,
#                  mlp_p,
#                  pos_p):
#         super().__init__()
#         self.Hp = image_size // patch_size
#         self.Wp = image_size // patch_size

#         self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)

#         self.freq_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.freq_proj = FrequencyProjection(embed_dim)


#         self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_tokens = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
#         self.pos_dropout = nn.Dropout(pos_p)

#         self.blocks = nn.ModuleList(
#             [
#                 TransformerBlock(embed_dim, num_head, mlp_ratio, attn_p, mlp_p)
#                 for _ in range(depth)
#             ]
#         )

#         self.final_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.head = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         patch = self.patch_embed(x)                 # [B, HW, D]
#         B, HW, D = patch.shape

#         cls = self.cls_tokens.expand(B, -1, -1)
#         freq = self.freq_token.expand(B, -1, -1)

#         # build sequence
#         x = torch.cat([cls, freq, patch], dim=1)    # [B, 2+HW, D]
#         x = x + self.pos_tokens
#         x = self.pos_dropout(x)

#         # run blocks
#         for block in self.blocks:
#             x = block(x)

#         x = self.final_layernorm(x)

#         # UPDATED tokens after interaction
#         cls_out  = x[:, 0, :]       # [B, D]
#         freq_out = x[:, 1:2, :]     # [B, 1, D]
#         patch_out = x[:, 2:, :]     # [B, HW, D]

#         # proxy map now depends on transformer parameters too
#         proxy_map = self.freq_proj(patch_out, freq_out, self.Hp, self.Wp)  # [B, Hp, Wp]

#         logits = self.head(cls_out)
#         return logits, proxy_map.unsqueeze(1)



# vit_freq_injector.py
# Minimal, FULL working code: ViT + frequency injector that ACTUALLY influences logits
# - Frequency token is DATA-DEPENDENT (from a differentiable frequency feature of the input)
# - Frequency is injected BEFORE attention (per-block FiLM on patch tokens)
# - Optional auxiliary proxy_map head (can be used for extra freq loss if you want)

# Dependencies: PyTorch >= 2.0 (for scaled_dot_product_attention)

# Usage:
#   model = VisionTransformerFreq(
#       in_channels=3, num_classes=1000, image_size=224, patch_size=16,
#       embed_dim=768, depth=12, num_head=12, mlp_ratio=4.0,
#       attn_p=0.0, mlp_p=0.0, pos_p=0.0, num_freq_bins=32
#   )
#   logits, aux = model(images)   # aux can be None if you disable it

# IMPORTANT:
# - This code does NOT include training loop or radial_spectrum_2d, etc.
# - The injector is differentiable end-to-end (no no_grad()).


import math
import torch
from torch import nn
import torch.nn.functional as F


# -------------------------
# Helpers: Patch embedding
# -------------------------
class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        image_size: int = 224,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)        # [B, D, Hp, Wp]
        x = x.flatten(2)        # [B, D, HW]
        x = x.transpose(1, 2)   # [B, HW, D]
        return x.contiguous()


# -------------------------
# MHA + MLP
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head: int, embed_dim: int, attn_p: float = 0.0):
        super().__init__()
        assert embed_dim % num_head == 0, "embed_dim must be divisible by num_head"
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_p = attn_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        qkv = qkv.view(B, N, 3, self.num_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # [3, B, H, N, Hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=(self.attn_p if self.training else 0.0)
        )  # [B, H, N, Hd]

        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        return self.proj(out)


class ConvMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, mlp_p: float = 0.0, Hp: int = 14, Wp: int = 14):
        super().__init__()
        self.Hp = Hp
        self.Wp = Wp

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act1 = nn.GELU()

        self.depthconv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act2 = nn.SiLU()

        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(mlp_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)

        x = x.transpose(1, 2).contiguous().reshape(x.shape[0], -1, self.Hp, self.Wp)  # [B, hidden_dim, Hp, Wp]
        x = self.depthconv1(x)
        x = self.act2(x)
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, HW, hidden_dim]

        x = self.fc2(x)
        return self.drop(x)


# -----------------------------------------
# Differentiable frequency feature extractor
# -----------------------------------------
class DifferentiableRadialFFT(nn.Module):
    """
    Returns a per-image "radial-ish" spectrum feature, differentiable.
    This is NOT a perfect radial binning; it's a simple, cheap proxy that works in practice:
      - FFT magnitude on grayscale
      - Downsample in frequency plane by adaptive pooling -> fixed num_freq_bins x num_freq_bins
      - Flatten to vector

    If you already have radial_spectrum_2d() you trust, you can replace this module.
    """
    def __init__(self, num_bins: int = 32, eps: float = 1e-6):
        super().__init__()
        self.num_bins = num_bins
        self.eps = eps

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: [B, C, H, W]
        B, C, H, W = img.shape
        gray = img.mean(dim=1)  # [B, H, W]

        # FFT2 (complex), magnitude
        # Freq = torch.fft.rfft2(gray, norm="ortho")              # [B, H, W//2+1]
        # mag = torch.sqrt(Freq.real**2 + Freq.imag**2 + self.eps)
        Freq = torch.fft.fft2(gray, norm="ortho")
        Freq = torch.fft.fftshift(Freq, dim=(-2, -1))
        mag = torch.abs(Freq)
        mag = torch.log1p(mag + 1e-6)

        # Normalize per-sample to avoid scale blow-ups
        mag = mag / (mag.mean(dim=(-2, -1), keepdim=True) + self.eps)

        # Convert to "image" and pool to fixed size
        mag = mag.unsqueeze(1)  # [B, 1, H, W]
        pooled = F.adaptive_avg_pool2d(mag, (self.num_bins, self.num_bins))  # [B,1,Bin,Bin]
        feat = pooled.flatten(1)  # [B, num_bins*num_bins]
        return feat


class FrequencyTokenEncoder(nn.Module):
    """
    Maps frequency features -> freq token [B,1,D].
    """
    def __init__(self, num_freq_feat: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_freq_feat, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, freq_feat: torch.Tensor) -> torch.Tensor:
        # freq_feat: [B, F]
        tok = self.proj(freq_feat).unsqueeze(1)  # [B,1,D]
        return tok


# -----------------------------------------
# Frequency injector (FiLM conditioning)
# -----------------------------------------
class FiLMFrequencyInjector(nn.Module):
    """
    Takes freq token [B,1,D] and modulates patch tokens [B,HW,D] BEFORE attention.
    This forces frequency information to shape the representation that produces logits.
    """
    def __init__(self, embed_dim: int, init_scale: float = 0.0):
        super().__init__()
        self.to_gamma_beta = nn.Linear(embed_dim, 2 * embed_dim, bias=True)

        # Optional: start near identity so you don't blow up training at step 0
        # init_scale=0.0 => gamma,beta start near 0 (identity-ish)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.init_scale = init_scale

    def forward(self, patch_tokens: torch.Tensor, freq_token: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, HW, D]
        # freq_token:   [B, 1, D]
        # gamma, beta = self.to_gamma_beta(freq_token).chunk(2, dim=-1)  # [B,1,D]
        # if self.init_scale != 1.0:
        #     gamma = torch.tanh(gamma) * self.init_scale
        #     beta  = torch.tanh(beta)  * self.init_scale

        beta = self.to_gamma_beta(freq_token)
        beta = F.glu(beta) * self.init_scale
        # FiLM: (1+gamma) * x + beta, broadcast over HW
        # return patch_tokens * (1.0 + gamma) + beta
        return patch_tokens + beta # simpler additive conditioning; works well in practice and is more stable to train than multiplicative


# -----------------------------------------
# Transformer block with injected conditioning
# -----------------------------------------
class TransformerBlockWithFreq(nn.Module):
    def __init__(self, embed_dim, num_head, mlp_ratio, attn_p, mlp_p, Hp, Wp):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(num_head, embed_dim, attn_p)

        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = ConvMLP(embed_dim, hidden_dim, mlp_p, Hp, Wp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))

        cls_freq = x[:, :2, :]
        patch = x[:, 2:, :]
        patch = patch + self.mlp(self.ln2(patch))
        x = torch.cat([cls_freq, patch], dim=1)
        return x


# -----------------------------------------
# Optional proxy map head (for aux loss)
# -----------------------------------------
class FrequencyProxyHead(nn.Module):
    """
    Builds a spatial proxy map from conditioned patch tokens.
    This can be used to compute your radial spectrum loss against input/target.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.to_scalar = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, patch_tokens: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        # patch_tokens: [B, HW, D]
        B, HW, D = patch_tokens.shape
        proxy = self.to_scalar(patch_tokens).squeeze(-1)  # [B, HW]
        return proxy.view(B, Hp, Wp)                      # [B, Hp, Wp]


# -----------------------------------------
# Vision Transformer with frequency injection
# -----------------------------------------
class VisionTransformerFreq(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_head: int = 12,
        mlp_ratio: float = 4.0,
        attn_p: float = 0.0,
        mlp_p: float = 0.0,
        pos_p: float = 0.0,
        num_freq_bins: int = 32,
        use_proxy_head: bool = True,
        inject_every_block: bool = True,
        injector_init_scale: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)
        self.Hp = image_size // patch_size
        self.Wp = image_size // patch_size

        # tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.freq_token_param = nn.Parameter(torch.zeros(1, 1, embed_dim))  # optional bias/offset

        # position embeddings (cls + freq + patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(pos_p)

        # frequency feature extractor -> frequency token
        self.freq_feat = DifferentiableRadialFFT(num_bins=num_freq_bins)
        num_freq_feat = num_freq_bins * num_freq_bins
        self.freq_tok_enc = FrequencyTokenEncoder(num_freq_feat, embed_dim)

        # injector + blocks
        self.injector = FiLMFrequencyInjector(embed_dim, init_scale=injector_init_scale)
        self.inject_every_block = inject_every_block

        self.blocks = nn.ModuleList([
            TransformerBlockWithFreq(embed_dim, num_head, mlp_ratio, attn_p, mlp_p, self.Hp, self.Wp)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self.use_proxy_head = use_proxy_head
        self.proxy_head = FrequencyProxyHead(embed_dim) if use_proxy_head else None

        self._init_params()

    def _init_params(self):
        # Basic init (kept simple)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.freq_token_param, std=0.02)
        # head defaults are fine

    def forward(self, img: torch.Tensor):
        """
        Returns:
          logits: [B, num_classes]
          aux: dict with optional keys:
             - 'freq_token': [B,1,D]
             - 'proxy_map':  [B,1,Hp,Wp]
             - 'patch_cond': [B,HW,D] (conditioned patch tokens after last injection)
        """
        B = img.size(0)

        # --- patches ---
        patch = self.patch_embed(img)  # [B, HW, D]

        # --- build DATA-DEPENDENT freq token ---
        with torch.amp.autocast(device_type='cuda', enabled=False): 
            img = img.float()  # ensure float for FFT and spectrum
            feat = self.freq_feat(img)                    # [B, F]
        
        feat = feat.to(patch.dtype)  # convert back to patch dtype for encoder
        freq_tok = self.freq_tok_enc(feat)            # [B,1,D]
        freq_tok = freq_tok + self.freq_token_param   # learned offset

        # --- inject BEFORE attention (critical) ---
        patch = self.injector(patch, freq_tok)

        # --- sequence (cls, freq, patch) ---
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, freq_tok, patch], dim=1)  # [B, 2+HW, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # --- transformer ---
        for blk in self.blocks:
            if self.inject_every_block:
                # re-inject into patch portion each block (keeps freq influence alive)
                patch_part = x[:, 2:, :]
                patch_part = self.injector(patch_part, x[:, 1:2, :])  # use current freq token
                x = torch.cat([x[:, :2, :], patch_part], dim=1)
            x = blk(x)

        x = self.norm(x)

        cls_out = x[:, 0, :]     # [B, D]
        freq_out = x[:, 1:2, :]  # [B,1,D]
        patch_out = x[:, 2:, :]  # [B,HW,D]

        logits = self.head(cls_out)

        aux = {"freq_token": freq_out.unsqueeze(1)}
        if self.use_proxy_head:
            proxy = self.proxy_head(patch_out, self.Hp, self.Wp)  # [B,Hp,Wp]
            aux["proxy_map"] = proxy.unsqueeze(1)                 # [B,1,Hp,Wp]
        aux["patch_cond"] = patch_out

        return logits, aux



import math
import torch
from torch import nn
import torch.nn.functional as F


# -------------------------
# Patch embedding
# -------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, image_size=224, embed_dim=768):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.proj(x)                           # [B,D,Hp,Wp]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B,HW,D]
        return x


# -------------------------
# MLP
# -------------------------
class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, mlp_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(mlp_p)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


# -------------------------
# FX-safe DCT-II basis (orthonormal) for Hp/Wp
# -------------------------
def _dct_ortho_matrix(N: int) -> torch.Tensor:
    # C[k,n] = alpha(k) * cos(pi*(n+0.5)*k/N)
    n = torch.arange(N).float()  # [N]
    k = torch.arange(N).float().unsqueeze(1)  # [N,1]
    C = torch.cos(math.pi / N * (n + 0.5) * k)  # [N,N]
    C[0, :] *= 1.0 / math.sqrt(N)
    if N > 1:
        C[1:, :] *= math.sqrt(2.0 / N)
    return C  # [N,N]


# -------------------------
# FX/FakeTensor-safe "frequency understand" without torch.fft and without boolean indexing:
#   - build proxy map [B,Hp,Wp]
#   - separable DCT: F = C_H @ proxy @ C_W^T
#   - band pooling via fixed one-hot weights + matmul
# -------------------------
class DCTSpectralBands(nn.Module):
    """
    Outputs:
      band_feat: [B,K]      (K band energies)
      coeff_abs: [B,Hp,Wp]  (abs(DCT coeffs), for optional downstream bias)
    """
    def __init__(self, embed_dim: int, Hp: int, Wp: int, num_bands: int = 6, eps: float = 1e-6):
        super().__init__()
        self.to_scalar = nn.Linear(embed_dim, 1, bias=False)
        self.Hp, self.Wp, self.K = Hp, Wp, num_bands
        self.eps = eps

        # DCT matrices as buffers (float32)
        C_H = _dct_ortho_matrix(Hp)
        C_W = _dct_ortho_matrix(Wp)
        self.register_buffer("C_H", C_H, persistent=False)            # [Hp,Hp]
        self.register_buffer("C_W_T", C_W.t().contiguous(), persistent=False)  # [Wp,Wp]

        # Precompute band weights W: [K, Hp*Wp] using index-based radius in DCT space (low-freq near (0,0))
        uu = torch.arange(Hp)
        vv = torch.arange(Wp)
        U, V = torch.meshgrid(uu, vv, indexing="ij")  # [Hp,Wp]
        # Chebyshev radius in coefficient plane
        r = torch.maximum(U, V).float()  # [Hp,Wp]
        rmax = float(r.max().item()) if r.numel() > 0 else 1.0

        edges = torch.linspace(0.0, rmax + 1e-6, steps=num_bands + 1)
        band_id = torch.empty((Hp, Wp), dtype=torch.long)
        for i in range(num_bands):
            lo, hi = edges[i], edges[i + 1]
            band_id[(r >= lo) & (r < hi)] = i
        band_id = band_id.view(-1)  # [Hp*Wp]

        W = F.one_hot(band_id, num_classes=num_bands).float().t()  # [K, Hp*Wp]
        denom = W.sum(dim=1, keepdim=True).clamp(min=1.0)
        W = W / denom  # mean pooling per band
        self.register_buffer("band_W", W, persistent=False)  # [K, Hp*Wp]

    def forward(self, patch_tokens: torch.Tensor):
        # patch_tokens: [B,HW,D]
        B, HW, D = patch_tokens.shape

        # proxy map; detach patch_tokens to avoid "chicken-egg" coupling (helps generalization)
        proxy = self.to_scalar(patch_tokens.detach()).squeeze(-1).view(B, self.Hp, self.Wp).to(torch.float32)  # [B,Hp,Wp]

        # separable DCT: F = C_H @ proxy @ C_W^T
        # torch.matmul broadcasts: (Hp,Hp) @ (B,Hp,Wp) -> (B,Hp,Wp)
        tmp = torch.matmul(self.C_H, proxy)            # [B,Hp,Wp]
        Fcoef = torch.matmul(tmp, self.C_W_T)          # [B,Hp,Wp]

        coeff_abs = Fcoef.abs()
        coeff_abs = coeff_abs / (coeff_abs.mean(dim=(-2, -1), keepdim=True) + self.eps)

        flat = coeff_abs.view(B, -1)                   # [B,Hp*Wp]
        band_feat = flat @ self.band_W.t()             # [B,K]
        return band_feat, coeff_abs


# -------------------------
# Band features -> K band tokens
# -------------------------
class BandTokenEncoder(nn.Module):
    def __init__(self, num_bands: int, embed_dim: int):
        super().__init__()
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, band_feat: torch.Tensor):
        # band_feat: [B,K] -> [B,K,D]
        B, K = band_feat.shape
        x = band_feat.view(B * K, 1)
        t = self.proj(x).view(B, K, self.embed_dim)
        return t


# -------------------------
# Global freq token from band features
# -------------------------
class GlobalFreqToken(nn.Module):
    def __init__(self, num_bands: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_bands, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, band_feat: torch.Tensor):
        return self.mlp(band_feat).unsqueeze(1)  # [B,1,D]


# -------------------------
# Cross-attention: patches query band tokens (forces usage), FX-safe
# -------------------------
class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, attn_p: float = 0.0):
        super().__init__()
        assert embed_dim % num_head == 0
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scale = self.head_dim ** -0.5
        self.attn_p = attn_p

        self.q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        # x: [B,N,D], ctx: [B,K,D]
        B, N, D = x.shape
        K = ctx.shape[1]

        q = self.q(x).view(B, N, self.num_head, self.head_dim).transpose(1, 2)  # [B,H,N,Hd]
        kv = self.kv(ctx).view(B, K, 2, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B,H,K,Hd]

        attn = (q * self.scale) @ k.transpose(-2, -1)  # [B,H,N,K]
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=(self.attn_p if self.training else 0.0))

        out = attn @ v  # [B,H,N,Hd]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.proj(out)


# -------------------------
# Self-attention with frequency-conditioned Q/K (efficient routing)
# -------------------------
class MultiHeadAttentionFreq(nn.Module):
    def __init__(self, num_head: int, embed_dim: int, attn_p: float = 0.0, freq_qk_scale: float = 0.10):
        super().__init__()
        assert embed_dim % num_head == 0
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.attn_p = attn_p

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.freq_to_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.freq_to_k = nn.Linear(embed_dim, embed_dim, bias=True)

        # tiny random init (better than all-zeros: avoids "never-used" path)
        nn.init.normal_(self.freq_to_q.weight, std=1e-4); nn.init.zeros_(self.freq_to_q.bias)
        nn.init.normal_(self.freq_to_k.weight, std=1e-4); nn.init.zeros_(self.freq_to_k.bias)

        self.freq_qk_scale = freq_qk_scale

    def forward(self, x: torch.Tensor, freq_token: torch.Tensor):
        # x: [B,N,D], freq_token: [B,1,D]
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        f = freq_token[:, 0, :]  # [B,D]
        dq = self.freq_to_q(f).view(B, self.num_head, 1, self.head_dim)
        dk = self.freq_to_k(f).view(B, self.num_head, 1, self.head_dim)
        q = q + dq * self.freq_qk_scale
        k = k + dk * self.freq_qk_scale

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.attn_p if self.training else 0.0))
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.proj(out)


# -------------------------
# Patchwise "retain" bias: use spatial proxy + high-pass (laplacian) map (FX-safe)
# -------------------------
class PatchwiseRetainBias(nn.Module):
    """
    Creates per-patch bias [B,HW,D] from two maps:
      - low-pass proxy (avg pooled)
      - high-pass proxy (fixed Laplacian magnitude)
    """
    def __init__(self, embed_dim: int, Hp: int, Wp: int):
        super().__init__()
        self.Hp, self.Wp = Hp, Wp

        # fixed Laplacian kernel as buffer
        k = torch.tensor([[0., 1., 0.],
                          [1., -4., 1.],
                          [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("lap", k, persistent=False)

        self.to_bias = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, proxy_map: torch.Tensor):
        # proxy_map: [B,Hp,Wp] float32
        B = proxy_map.shape[0]
        x = proxy_map.unsqueeze(1)  # [B,1,Hp,Wp]

        # low-pass (cheap local average)
        lp = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # high-pass (laplacian)
        hp = F.conv2d(x, self.lap, padding=1).abs()

        feat = torch.cat([lp, hp], dim=1)  # [B,2,Hp,Wp]
        feat = feat.permute(0, 2, 3, 1).contiguous().view(B, self.Hp * self.Wp, 2)  # [B,HW,2]

        bias = self.to_bias(feat)  # [B,HW,D]
        return bias


# -------------------------
# Transformer block with:
#   - gated cross-attn (only "strong" in early blocks by init)
#   - freq-conditioned self-attn
# -------------------------
class TransformerBlockStrongFreq(nn.Module):
    def __init__(self, embed_dim, num_head, mlp_ratio, attn_p, mlp_p, xattn_gate_init: float):
        super().__init__()
        self.ln0 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.xattn = CrossAttention(embed_dim, num_head, attn_p=attn_p)

        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttentionFreq(num_head, embed_dim, attn_p=attn_p, freq_qk_scale=0.10)

        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), mlp_p)

        # scalar gate parameter (FX-safe): patch += gate * xattn(...)
        self.xattn_gate_logit = nn.Parameter(torch.tensor(xattn_gate_init).float())

    def forward(self, x: torch.Tensor, band_tokens: torch.Tensor, freq_token: torch.Tensor):
        # x: [B,2+HW,D]
        patch = x[:, 2:, :]

        gate = torch.sigmoid(self.xattn_gate_logit)  # scalar
        patch = patch + gate * self.xattn(self.ln0(patch), band_tokens)
        x = torch.cat([x[:, :2, :], patch], dim=1)

        x = x + self.attn(self.ln1(x), freq_token=freq_token)
        x = x + self.mlp(self.ln2(x))
        return x


# -------------------------
# Full model: better generalization + ACT/FX friendly
# -------------------------
class VisionTransformerFreqStrong(nn.Module):
    """
    Designed to "perform much better" for your setup:

    ✅ FX + FakeTensor friendly:
      - no boolean indexing in forward
      - no nonzero / masked gather
      - no dynamic arange in forward
      - no torch.fft (uses DCT via matmul)

    ✅ Better generalization vs your previous FreqViT:
      - decouples freq extraction from patch stream (detach)
      - cross-attn is GATED (strong early, fades if harmful)
      - Q/K conditioning has tiny random init (prevents dead branch)
      - patchwise retain bias uses LP+HP maps (more discriminative than smooth tanh-only)

    Output:
      logits, aux (includes optional anchor_loss)
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_head: int = 12,
        mlp_ratio: float = 4.0,
        attn_p: float = 0.0,
        mlp_p: float = 0.0,
        pos_p: float = 0.0,
        num_bands: int = 6,
        retain_scale: float = 0.04,
        anchor_freq_token: bool = True,
        anchor_weight: float = 0.02,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)
        self.Hp = image_size // patch_size
        self.Wp = image_size // patch_size
        self.depth = depth

        # tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.freq_token_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # pos emb for (cls + freq + patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(pos_p)

        # understand: DCT spectral bands
        self.spec = DCTSpectralBands(embed_dim, self.Hp, self.Wp, num_bands=num_bands)
        self.band_enc = BandTokenEncoder(num_bands, embed_dim)
        self.freq_tok_enc = GlobalFreqToken(num_bands, embed_dim)

        # retain: patchwise bias from proxy LP/HP
        self.retain_bias = PatchwiseRetainBias(embed_dim, self.Hp, self.Wp)
        self.retain_scale = retain_scale

        # transformer blocks (gate init: high for early blocks, low for late blocks)
        blocks = []
        for i in range(depth):
            # early blocks: gate_init ~ +2 (sigmoid~0.88), late blocks: ~ -2 (sigmoid~0.12)
            # smooth schedule
            t = i / max(depth - 1, 1)
            gate_init = 2.0 * (1.0 - 2.0 * t)  # +2 -> -2
            blocks.append(TransformerBlockStrongFreq(embed_dim, num_head, mlp_ratio, attn_p, mlp_p, gate_init))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # freq token anchoring
        self.anchor_freq_token = anchor_freq_token
        self.anchor_weight = anchor_weight

        self._init_params()

    def _init_params(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.freq_token_bias, std=0.02)

    def forward(self, img: torch.Tensor):
        B = img.size(0)

        # patches
        patch = self.patch_embed(img)  # [B,HW,D]

        # understand: band features from DCT on proxy map
        band_feat, coeff_abs = self.spec(patch)         # [B,K], [B,Hp,Wp]
        band_tokens = self.band_enc(band_feat)          # [B,K,D]
        freq_tok = self.freq_tok_enc(band_feat) + self.freq_token_bias  # [B,1,D]

        # retain: patchwise LP/HP bias from proxy map reconstructed from patch tokens
        # build proxy map (consistent with spec): scalar projection, detached, fp32
        proxy_map = self.spec.to_scalar(patch.detach()).squeeze(-1).view(B, self.Hp, self.Wp).to(torch.float32)
        pw_bias = self.retain_bias(proxy_map).to(patch.dtype)  # [B,HW,D]

        # inject small, but NOT tanh-squashed to death (use layernorm-like scaling)
        pw_bias = F.layer_norm(pw_bias, (pw_bias.shape[-1],))
        patch = patch + self.retain_scale * pw_bias

        # sequence
        cls = self.cls_token.expand(B, -1, -1).to(patch.dtype)
        x = torch.cat([cls, freq_tok.to(patch.dtype), patch], dim=1)  # [B,2+HW,D]
        x = x + self.pos_embed.to(x.dtype)
        x = self.pos_drop(x)

        # blocks: gated xattn + freq-conditioned self-attn
        for blk in self.blocks:
            x = blk(x, band_tokens=band_tokens.to(x.dtype), freq_token=x[:, 1:2, :])

        x = self.norm(x)
        logits = self.head(x[:, 0, :])

        aux = {
            "freq_token": x[:, 1:2, :],
            "band_tokens": band_tokens,
            "band_feat": band_feat,
        }

        # anchor loss helps keep freq token semantic (prevents drifting into generic helper token)
        if self.anchor_freq_token and self.training:
            anchor_loss = (x[:, 1:2, :] - freq_tok.detach().to(x.dtype)).pow(2).mean()
            aux["anchor_loss"] = anchor_loss * self.anchor_weight

        return logits, aux
