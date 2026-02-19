# import math
# import torch
# from torch import nn
# import torch.nn.functional as F


# # -------------------------
# # Patch embedding
# # -------------------------
# class PatchEmbed(nn.Module):
#     def __init__(
#         self,
#         in_channels: int = 3,
#         patch_size: int = 16,
#         image_size: int = 224,
#         embed_dim: int = 768,
#     ):
#         super().__init__()
#         assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
#         self.patch_size = patch_size
#         self.image_size = image_size
#         self.grid = image_size // patch_size
#         self.num_patches = self.grid * self.grid

#         self.proj = nn.Conv2d(
#             in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, C, H, W]
#         x = self.proj(x)        # [B, D, Hp, Wp]
#         x = x.flatten(2)        # [B, D, HW]
#         x = x.transpose(1, 2)   # [B, HW, D]
#         return x.contiguous()


# # -------------------------
# # MLP
# # -------------------------
# class MLP(nn.Module):
#     def __init__(self, embed_dim: int, hidden_dim: int, mlp_p: float = 0.0):
#         super().__init__()
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, embed_dim)
#         self.drop = nn.Dropout(mlp_p)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         return self.drop(x)


# # -----------------------------------------
# # Patch-proxy frequency features (better aligned than raw-image FFT)
# # -----------------------------------------
# class DifferentiableFreqFromPatchProxy(nn.Module):
#     """
#     Build a frequency feature from a patch-grid proxy map:
#       - patch_tokens -> scalar proxy per patch -> [B, Hp, Wp]
#       - FFT2 on proxy (small grid, stable)
#       - log magnitude + per-sample normalization
#       - adaptive pool -> fixed (num_bins x num_bins), flatten
#     """
#     def __init__(self, embed_dim: int, num_bins: int = 16, eps: float = 1e-6):
#         super().__init__()
#         self.num_bins = num_bins
#         self.eps = eps
#         self.to_scalar = nn.Linear(embed_dim, 1, bias=False)

#     def forward(self, patch_tokens: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
#         # patch_tokens: [B, HW, D]
#         B, HW, D = patch_tokens.shape
#         # assert HW == Hp * Wp, f"HW mismatch: {HW} vs Hp*Wp {Hp*Wp}"

#         proxy = self.to_scalar(patch_tokens).squeeze(-1)  # [B, HW]
#         proxy = proxy.view(B, Hp, Wp).to(torch.float32)   # do FFT in fp32 for stability

#         Freq = torch.fft.fft2(proxy, norm="ortho")
#         mag = torch.abs(Freq)
#         mag = torch.fft.fftshift(mag, dim=(-2, -1))
#         mag = torch.log1p(mag + 1e-6)

#         # normalize to avoid scale blowups
#         mag = mag / (mag.mean(dim=(-2, -1), keepdim=True) + self.eps)

#         mag = mag.unsqueeze(1)  # [B, 1, Hp, Wp]
#         pooled = F.adaptive_avg_pool2d(mag, (self.num_bins, self.num_bins))  # [B,1,b,b]
#         feat = pooled.flatten(1)  # [B, b*b]
#         return feat


# class FrequencyTokenEncoder(nn.Module):
#     """
#     Maps frequency features -> freq token [B,1,D].
#     """
#     def __init__(self, num_freq_feat: int, embed_dim: int):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(num_freq_feat, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim),
#         )

#     def forward(self, freq_feat: torch.Tensor) -> torch.Tensor:
#         tok = self.proj(freq_feat).unsqueeze(1)  # [B,1,D]
#         return tok


# # -----------------------------------------
# # Frequency-conditioned attention (efficient use: biases Q/K)
# # -----------------------------------------
# class MultiHeadAttentionFreq(nn.Module):
#     """
#     Standard MHA + optional conditioning:
#       q += Wq_f(freq_token)
#       k += Wk_f(freq_token)

#     This makes attention routing directly frequency-aware (usually the biggest win).
#     """
#     def __init__(self, num_head: int, embed_dim: int, attn_p: float = 0.0, freq_qk_scale: float = 0.05):
#         super().__init__()
#         assert embed_dim % num_head == 0, "embed_dim must be divisible by num_head"
#         self.num_head = num_head
#         self.head_dim = embed_dim // num_head
#         self.attn_p = attn_p

#         self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
#         self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

#         # freq -> q/k biases
#         self.freq_to_q = nn.Linear(embed_dim, embed_dim, bias=True)
#         self.freq_to_k = nn.Linear(embed_dim, embed_dim, bias=True)

#         # start near 0 so step-0 is basically baseline ViT
#         nn.init.zeros_(self.freq_to_q.weight); nn.init.zeros_(self.freq_to_q.bias)
#         nn.init.zeros_(self.freq_to_k.weight); nn.init.zeros_(self.freq_to_k.bias)

#         self.freq_qk_scale = freq_qk_scale

#     def forward(self, x: torch.Tensor, freq_token: torch.Tensor | None = None) -> torch.Tensor:
#         # x: [B, N, D]
#         B, N, D = x.shape

#         qkv = self.qkv(x)  # [B, N, 3D]
#         qkv = qkv.view(B, N, 3, self.num_head, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # [3, B, H, N, Hd]
#         q, k, v = qkv[0], qkv[1], qkv[2]               # [B, H, N, Hd]

#         if freq_token is not None:
#             # freq_token: [B,1,D] or [B,D]
#             f = freq_token[:, 0, :]  # [B, D]

#             dq = self.freq_to_q(f).view(B, self.num_head, 1, self.head_dim)  # [B,H,1,Hd]
#             dk = self.freq_to_k(f).view(B, self.num_head, 1, self.head_dim)

#             # scaled, broadcast over tokens
#             q = q + dq * self.freq_qk_scale
#             k = k + dk * self.freq_qk_scale

#         out = F.scaled_dot_product_attention(
#             q, k, v, dropout_p=(self.attn_p if self.training else 0.0)
#         )  # [B, H, N, Hd]

#         out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
#         return self.proj(out)


# # -----------------------------------------
# # Small residual conditioning (optional, stable)
# # -----------------------------------------
# class AdditiveFrequencyInjector(nn.Module):
#     """
#     Adds a per-channel bias to all patch tokens from freq token.
#     Kept small & gated to avoid overpowering CE.
#     """
#     def __init__(self, embed_dim: int, scale: float = 0.1):
#         super().__init__()
#         self.to_beta = nn.Linear(embed_dim, embed_dim, bias=True)
#         nn.init.zeros_(self.to_beta.weight)
#         nn.init.zeros_(self.to_beta.bias)
#         self.scale = scale

#     def forward(self, patch_tokens: torch.Tensor, freq_token: torch.Tensor) -> torch.Tensor:
#         # patch_tokens: [B, HW, D], freq_token: [B,1,D]
#         beta = self.to_beta(freq_token)          # [B,1,D]
#         beta = torch.tanh(beta) * self.scale     # bounded
#         return patch_tokens + beta


# # -----------------------------------------
# # Transformer block (freq-conditioned attention + optional reinjection)
# # -----------------------------------------
# class TransformerBlockFreq(nn.Module):
#     def __init__(self, embed_dim, num_head, mlp_ratio, attn_p, mlp_p):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.attn = MultiHeadAttentionFreq(num_head, embed_dim, attn_p, freq_qk_scale=0.05)

#         self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
#         hidden_dim = int(embed_dim * mlp_ratio)
#         self.mlp = MLP(embed_dim, hidden_dim, mlp_p)

#     def forward(self, x: torch.Tensor, freq_token: torch.Tensor) -> torch.Tensor:
#         x = x + self.attn(self.ln1(x), freq_token=freq_token)
#         x = x + self.mlp(self.ln2(x))

#         print("freq_to_q absmax:", self.attn.freq_to_q.weight.abs().max())
#         print("freq_token absmax:", freq_token.abs().max())

#         return x


# # -----------------------------------------
# # Optional proxy head for aux loss
# # -----------------------------------------
# class FrequencyProxyHead(nn.Module):
#     def __init__(self, embed_dim: int):
#         super().__init__()
#         self.to_scalar = nn.Linear(embed_dim, 1, bias=False)

#     def forward(self, patch_tokens: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
#         B, HW, D = patch_tokens.shape
#         proxy = self.to_scalar(patch_tokens).squeeze(-1)  # [B, HW]
#         return proxy.view(B, Hp, Wp)                      # [B, Hp, Wp]


# # -----------------------------------------
# # Vision Transformer with efficient frequency usage
# # -----------------------------------------
# class VisionTransformerFreqV2(nn.Module):
#     """
#     Key upgrades vs your current version:
#       1) Frequency feature is computed from patch-proxy (aligned with ViT tokens)
#       2) Attention is frequency-conditioned via Q/K biases (harder to ignore => more efficient usage)
#       3) Optional small residual reinjection per block (bounded) to keep freq influence alive
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         num_classes: int,
#         image_size: int = 224,
#         patch_size: int = 16,
#         embed_dim: int = 768,
#         depth: int = 12,
#         num_head: int = 12,
#         mlp_ratio: float = 4.0,
#         attn_p: float = 0.0,
#         mlp_p: float = 0.0,
#         pos_p: float = 0.0,
#         freq_bins: int = 16,                 # smaller is usually enough for a control signal
#         inject_every_block: bool = True,
#         injector_scale: float = 0.05,        # keep small under quantization
#         use_proxy_head: bool = True,
#     ):
#         super().__init__()
#         assert image_size % patch_size == 0
#         self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)
#         self.Hp = image_size // patch_size
#         self.Wp = image_size // patch_size

#         # tokens
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.freq_token_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))  # learned offset

#         # pos emb for (cls + freq + patches)
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
#         self.pos_drop = nn.Dropout(pos_p)

#         # freq feature from patch proxy
#         self.freq_feat = DifferentiableFreqFromPatchProxy(embed_dim=embed_dim, num_bins=freq_bins)
#         num_freq_feat = freq_bins * freq_bins
#         self.freq_tok_enc = FrequencyTokenEncoder(num_freq_feat, embed_dim)

#         # optional additive reinjection (bounded)
#         self.injector = AdditiveFrequencyInjector(embed_dim, scale=injector_scale)
#         self.inject_every_block = inject_every_block

#         self.blocks = nn.ModuleList([
#             TransformerBlockFreq(embed_dim, num_head, mlp_ratio, attn_p, mlp_p)
#             for _ in range(depth)
#         ])

#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.head = nn.Linear(embed_dim, num_classes)

#         self.use_proxy_head = use_proxy_head
#         self.proxy_head = FrequencyProxyHead(embed_dim) if use_proxy_head else None

#         self._init_params()

#     def _init_params(self):
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         nn.init.trunc_normal_(self.freq_token_bias, std=0.02)

#     def forward(self, img: torch.Tensor):
#         """
#         Returns:
#           logits: [B, num_classes]
#           aux: dict:
#             - 'freq_token': [B,1,D]
#             - 'proxy_map':  [B,1,Hp,Wp] (if use_proxy_head)
#             - 'patch_out':  [B,HW,D]
#         """
#         B = img.size(0)

#         # patch tokens
#         patch = self.patch_embed(img)  # [B, HW, D]

#         # build DATA-dependent freq token from patch-space proxy (stable)
#         # NOTE: FFT inside module runs in fp32; returns feat in fp32
#         feat = self.freq_feat(patch, self.Hp, self.Wp)          # [B, F] fp32
#         feat = feat.to(patch.dtype)                              # match main dtype
#         freq_tok = self.freq_tok_enc(feat) + self.freq_token_bias  # [B,1,D]

#         # optional initial injection to patch stream
#         patch = self.injector(patch, freq_tok)

#         # sequence: (cls, freq, patch)
#         cls = self.cls_token.expand(B, -1, -1)                   # [B,1,D]
#         x = torch.cat([cls, freq_tok, patch], dim=1)             # [B, 2+HW, D]
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         # transformer blocks
#         for blk in self.blocks:
#             # frequency-conditioned attention uses current freq token in the stream
#             x = blk(x, freq_token=x[:, 1:2, :])

#             if self.inject_every_block:
#                 # small bounded reinjection into patches to keep freq influence alive
#                 patch_part = x[:, 2:, :]
#                 patch_part = self.injector(patch_part, x[:, 1:2, :])
#                 x = torch.cat([x[:, :2, :], patch_part], dim=1)

#         x = self.norm(x)

#         cls_out = x[:, 0, :]       # [B,D]
#         freq_out = x[:, 1:2, :]    # [B,1,D]
#         patch_out = x[:, 2:, :]    # [B,HW,D]

#         logits = self.head(cls_out)

#         aux = {"freq_token": freq_out.unsqueeze(1), "patch_out": patch_out}
#         if self.use_proxy_head:
#             proxy = self.proxy_head(patch_out, self.Hp, self.Wp)    # [B,Hp,Wp]
#             aux["proxy_map"] = proxy.unsqueeze(1)                   # [B,1,Hp,Wp]

#         return logits, aux


# import torch
# from torch import nn
# import torch.nn.functional as F


# # -------------------------
# # Patch embedding
# # -------------------------
# class PatchEmbed(nn.Module):
#     def __init__(self, in_channels=3, patch_size=16, image_size=224, embed_dim=768):
#         super().__init__()
#         assert image_size % patch_size == 0
#         self.patch_size = patch_size
#         self.image_size = image_size
#         self.grid = image_size // patch_size
#         self.num_patches = self.grid * self.grid
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

#     def forward(self, x):
#         x = self.proj(x)          # [B,D,Hp,Wp]
#         x = x.flatten(2).transpose(1, 2).contiguous()  # [B,HW,D]
#         return x


# # -------------------------
# # MLP
# # -------------------------
# class MLP(nn.Module):
#     def __init__(self, embed_dim, hidden_dim, mlp_p=0.0):
#         super().__init__()
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden_dim, embed_dim)
#         self.drop = nn.Dropout(mlp_p)

#     def forward(self, x):
#         return self.drop(self.fc2(self.act(self.fc1(x))))


# # -------------------------
# # Patch-proxy FFT feature (global) + Band split (K tokens)
# # -------------------------
# class PatchProxySpectralBands(nn.Module):
#     def __init__(self, embed_dim, Hp, Wp, num_bands=6, eps=1e-6):
#         super().__init__()
#         self.to_scalar = nn.Linear(embed_dim, 1, bias=False)
#         self.Hp = Hp
#         self.Wp = Wp
#         self.K = num_bands
#         self.eps = eps

#         # ---- precompute band weights: [K, Hp*Wp] (float32, 0/1) ----
#         yy = torch.arange(Hp) - (Hp // 2)
#         xx = torch.arange(Wp) - (Wp // 2)
#         Y, X = torch.meshgrid(yy, xx, indexing="ij")
#         r = torch.maximum(Y.abs(), X.abs()).float()  # [Hp,Wp]
#         rmax = float(r.max().item()) if r.numel() > 0 else 1.0
#         edges = torch.linspace(0.0, rmax + 1e-6, steps=num_bands + 1)  # inclusive hi

#         band_id = torch.empty((Hp, Wp), dtype=torch.long)
#         for i in range(num_bands):
#             lo, hi = edges[i], edges[i + 1]
#             # IMPORTANT: avoid boolean indexing in forward; here it's OK in __init__
#             band_id[(r >= lo) & (r < hi)] = i
#         band_id = band_id.view(-1)  # [Hp*Wp]

#         W = F.one_hot(band_id, num_classes=num_bands).float().t()  # [K, Hp*Wp]
#         denom = W.sum(dim=1, keepdim=True).clamp(min=1.0)          # [K,1]
#         W = W / denom                                              # mean-pooling weights

#         self.register_buffer("band_W", W, persistent=False)         # [K, Hp*Wp]

#     def forward(self, patch_tokens):
#         B, HW, D = patch_tokens.shape
#         proxy = self.to_scalar(patch_tokens).squeeze(-1).view(B, self.Hp, self.Wp).to(torch.float32)

#         Freq = torch.fft.fft2(proxy, norm="ortho")
#         mag = torch.abs(Freq)
#         mag = torch.fft.fftshift(mag, dim=(-2, -1))
#         mag = torch.log1p(mag + 1e-6)
#         mag = mag / (mag.mean(dim=(-2, -1), keepdim=True) + self.eps)  # [B,Hp,Wp]

#         mag_flat = mag.view(B, -1)                         # [B, Hp*Wp]
#         band_feat = mag_flat @ self.band_W.t()             # [B, K]  (static matmul)

#         return band_feat, mag




# # -------------------------
# # Encode K band features -> K freq tokens
# # -------------------------
# class BandTokenEncoder(nn.Module):
#     """
#     band_feat: [B,K] -> band_tokens: [B,K,D]
#     """
#     def __init__(self, num_bands, embed_dim):
#         super().__init__()
#         self.num_bands = num_bands
#         self.embed_dim = embed_dim
#         self.proj = nn.Sequential(
#             nn.Linear(1, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim),
#         )

#     def forward(self, band_feat):
#         # band_feat: [B,K]
#         B, K = band_feat.shape
#         x = band_feat.view(B * K, 1)
#         t = self.proj(x).view(B, K, self.embed_dim)  # [B,K,D]
#         return t


# # -------------------------
# # Cross-attention: patches query band tokens (forces usage)
# # -------------------------
# class CrossAttention(nn.Module):
#     """
#     Q from x (patches), K/V from ctx (band tokens). FX-safe (no control flow).
#     """
#     def __init__(self, embed_dim, num_head, attn_p=0.0):
#         super().__init__()
#         assert embed_dim % num_head == 0
#         self.num_head = num_head
#         self.head_dim = embed_dim // num_head
#         self.scale = self.head_dim ** -0.5
#         self.attn_p = attn_p

#         self.q = nn.Linear(embed_dim, embed_dim, bias=True)
#         self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
#         self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

#     def forward(self, x, ctx):
#         # x:   [B,N,D]
#         # ctx: [B,K,D]
#         B, N, D = x.shape
#         K = ctx.shape[1]

#         q = self.q(x).view(B, N, self.num_head, self.head_dim).transpose(1, 2)        # [B,H,N,Hd]
#         kv = self.kv(ctx).view(B, K, 2, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]                                                           # [B,H,K,Hd]

#         attn = (q * self.scale) @ k.transpose(-2, -1)                                  # [B,H,N,K]
#         attn = attn.softmax(dim=-1)
#         attn = F.dropout(attn, p=(self.attn_p if self.training else 0.0))

#         out = attn @ v                                                                  # [B,H,N,Hd]
#         out = out.transpose(1, 2).contiguous().view(B, N, D)
#         return self.proj(out)


# # -------------------------
# # Patchwise frequency bias (per-patch, aligned signal)
# # -------------------------
# class PatchwiseFreqBias(nn.Module):
#     """
#     Creates a per-patch bias [B,HW,D] from the patch-grid FFT magnitude.
#     This is the "retain + localize" mechanism: frequency info becomes patch-aligned.
#     """
#     def __init__(self, embed_dim, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         self.to_bias = nn.Sequential(
#             nn.Linear(1, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim),
#         )

#     def forward(self, mag_grid, Hp, Wp):
#         # mag_grid: [B,Hp,Wp] (log-mag, normalized)
#         B = mag_grid.shape[0]
#         # downsample/upsample already aligned: it's Hp/Wp
#         m = mag_grid.view(B, Hp * Wp, 1)  # [B,HW,1]
#         bias = self.to_bias(m)            # [B,HW,D]
#         return bias


# # -------------------------
# # Self-attention with frequency-conditioned Q/K (efficient routing)
# # -------------------------
# class MultiHeadAttentionFreq(nn.Module):
#     def __init__(self, num_head, embed_dim, attn_p=0.0, freq_qk_scale=0.08):
#         super().__init__()
#         assert embed_dim % num_head == 0
#         self.num_head = num_head
#         self.head_dim = embed_dim // num_head
#         self.attn_p = attn_p

#         self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
#         self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

#         self.freq_to_q = nn.Linear(embed_dim, embed_dim, bias=True)
#         self.freq_to_k = nn.Linear(embed_dim, embed_dim, bias=True)
#         nn.init.zeros_(self.freq_to_q.weight); nn.init.zeros_(self.freq_to_q.bias)
#         nn.init.zeros_(self.freq_to_k.weight); nn.init.zeros_(self.freq_to_k.bias)

#         self.freq_qk_scale = freq_qk_scale

#     def forward(self, x, freq_token):
#         # x: [B,N,D], freq_token: [B,1,D]  (always, FX-safe)
#         B, N, D = x.shape

#         qkv = self.qkv(x).view(B, N, 3, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
#         q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,N,Hd]

#         f = freq_token[:, 0, :]  # [B,D]
#         dq = self.freq_to_q(f).view(B, self.num_head, 1, self.head_dim)
#         dk = self.freq_to_k(f).view(B, self.num_head, 1, self.head_dim)
#         q = q + dq * self.freq_qk_scale
#         k = k + dk * self.freq_qk_scale

#         out = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.attn_p if self.training else 0.0))
#         out = out.transpose(1, 2).contiguous().view(B, N, D)
#         return self.proj(out)


# # -------------------------
# # Transformer block: (1) forced cross-attn to band tokens + (2) freq-conditioned self-attn
# # -------------------------
# class TransformerBlockFullFreq(nn.Module):
#     def __init__(self, embed_dim, num_head, mlp_ratio, attn_p, mlp_p):
#         super().__init__()
#         self.ln0 = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.xattn = CrossAttention(embed_dim, num_head, attn_p=attn_p)  # patches query band tokens

#         self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.attn = MultiHeadAttentionFreq(num_head, embed_dim, attn_p=attn_p, freq_qk_scale=0.08)

#         self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), mlp_p)

#     def forward(self, x, band_tokens, freq_token):
#         # x: [B, 2+HW, D]
#         # band_tokens: [B,K,D]
#         # freq_token: [B,1,D]

#         # forced usage: patch part must read band tokens
#         patch = x[:, 2:, :]
#         patch = patch + self.xattn(self.ln0(patch), band_tokens)
#         x = torch.cat([x[:, :2, :], patch], dim=1)

#         # freq-conditioned self-attn (routing)
#         x = x + self.attn(self.ln1(x), freq_token=freq_token)

#         # MLP
#         x = x + self.mlp(self.ln2(x))
#         return x


# # -------------------------
# # Vision Transformer with "understand + retain" frequency mechanism
# # -------------------------
# class VisionTransformerFreqV2(nn.Module):
#     """
#     Fulfills your "freq understand + retain mechanism" requirements:

#     ✅ Understand:
#       - Extract patch-proxy FFT spectrum
#       - Split into K bands -> K band tokens
#     ✅ Retain:
#       - Patchwise frequency bias (per patch, aligned to tokens)
#       - Forced cross-attention each block: patches must read band tokens
#       - Frequency-conditioned self-attention: Q/K biased by freq token
#     ✅ FX-safe:
#       - No Proxy-dependent control flow
#       - freq_token always [B,1,D]
#     """
#     def __init__(
#         self,
#         in_channels,
#         num_classes,
#         image_size=224,
#         patch_size=16,
#         embed_dim=768,
#         depth=12,
#         num_head=12,
#         mlp_ratio=4.0,
#         attn_p=0.0,
#         mlp_p=0.0,
#         pos_p=0.0,
#         num_bands=6,
#         inject_patchwise_bias=True,
#         anchor_freq_token=True,
#         anchor_weight=0.02,   # small, safe
#         use_proxy_head=False, # kept off by default
#     ):
#         super().__init__()
#         assert image_size % patch_size == 0
#         self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)
#         self.Hp = image_size // patch_size
#         self.Wp = image_size // patch_size

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.freq_token_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))

#         self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
#         self.pos_drop = nn.Dropout(pos_p)

#         # understand: spectral bands -> band tokens
#         self.spec = PatchProxySpectralBands(embed_dim, self.Hp, self.Wp, num_bands=num_bands)
#         self.band_enc = BandTokenEncoder(num_bands=num_bands, embed_dim=embed_dim)

#         # global freq token from bands (simple: MLP over K)
#         self.freq_tok_enc = nn.Sequential(
#             nn.Linear(num_bands, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim),
#         )

#         # retain: patchwise bias from spectrum grid
#         self.inject_patchwise_bias = inject_patchwise_bias
#         self.patch_bias = PatchwiseFreqBias(embed_dim) if inject_patchwise_bias else None

#         # transformer
#         self.blocks = nn.ModuleList([
#             TransformerBlockFullFreq(embed_dim, num_head, mlp_ratio, attn_p, mlp_p)
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.head = nn.Linear(embed_dim, num_classes)

#         # optional: anchor freq token so it stays frequency-coded (prevents drifting)
#         self.anchor_freq_token = anchor_freq_token
#         self.anchor_weight = anchor_weight

#         self._init_params()

#     def _init_params(self):
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         nn.init.trunc_normal_(self.freq_token_bias, std=0.02)

#     def forward(self, img):
#         """
#         Returns:
#           logits
#           aux:
#             - freq_token: [B,1,D]
#             - band_tokens:[B,K,D]
#             - patch_bias: [B,HW,D] (if enabled)
#             - anchor_loss: scalar (if enabled, training-time)
#         """
#         B = img.size(0)

#         # patches
#         patch = self.patch_embed(img)  # [B,HW,D]

#         # understand: spectrum from patch proxy
#         band_feat, mag_grid = self.spec(patch)     # [B,K], [B,Hp,Wp]
#         band_tokens = self.band_enc(band_feat)                       # [B,K,D]

#         # global freq token from band features
#         freq_tok = self.freq_tok_enc(band_feat).unsqueeze(1)         # [B,1,D]
#         freq_tok = freq_tok + self.freq_token_bias                   # [B,1,D]

#         # retain: patchwise bias (aligned signal)
#         patch_bias = None
#         if self.inject_patchwise_bias:
#             patch_bias = self.patch_bias(mag_grid, self.Hp, self.Wp).to(patch.dtype)  # [B,HW,D]
#             patch = patch + 0.05 * torch.tanh(patch_bias)  # bounded, keep small

#         # sequence (cls, freq, patch)
#         cls = self.cls_token.expand(B, -1, -1)
#         x = torch.cat([cls, freq_tok.to(patch.dtype), patch], dim=1)  # [B,2+HW,D]
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         # blocks: forced cross-attn + freq-conditioned self-attn
#         for blk in self.blocks:
#             x = blk(x, band_tokens=band_tokens.to(x.dtype), freq_token=x[:, 1:2, :])

#         x = self.norm(x)
#         logits = self.head(x[:, 0, :])

#         aux = {
#             "freq_token": x[:, 1:2, :],
#             "band_tokens": band_tokens,
#         }
#         if patch_bias is not None:
#             aux["patch_bias"] = patch_bias

#         # anchor loss (returned in aux so your Controller can add it)
#         if self.anchor_freq_token and self.training:
#             # keep the in-stream freq token close to the originally encoded freq token
#             # detach target so it doesn't collapse the encoder; this prevents semantic drift
#             anchor_loss = (x[:, 1:2, :] - freq_tok.detach().to(x.dtype)).pow(2).mean()
#             aux["anchor_loss"] = anchor_loss * self.anchor_weight

#         return logits, aux


# import math
# from typing import Dict, Tuple

# import torch
# from torch import nn
# import torch.nn.functional as F


# # =========================================================
# # Utils
# # =========================================================
# def trunc_normal_(tensor, mean=0.0, std=0.02):
#     with torch.no_grad():
#         size = tensor.shape
#         tmp = tensor.new_empty(size + (4,)).normal_()
#         valid = (tmp < 2) & (tmp > -2)
#         ind = valid.max(-1, keepdim=True)[1]
#         tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
#         tensor.mul_(std).add_(mean)
#     return tensor


# # =========================================================
# # Patch embedding
# # =========================================================
# class PatchEmbed(nn.Module):
#     def __init__(self, in_channels=3, patch_size=16, image_size=224, embed_dim=768):
#         super().__init__()
#         assert image_size % patch_size == 0
#         self.grid = image_size // patch_size
#         self.num_patches = self.grid * self.grid
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

#     def forward(self, x):
#         x = self.proj(x)                               # [B,D,Hp,Wp]
#         x = x.flatten(2).transpose(1, 2).contiguous()  # [B,HW,D]
#         return x


# # =========================================================
# # MLP
# # =========================================================
# class ConvMLP(nn.Module):
#     def __init__(self, embed_dim, hidden_dim, Hp, Wp, mlp_p=0.0):
#         super().__init__()
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.act1 = nn.GELU()

#         self.dwconv = nn.Conv2d(
#             hidden_dim,
#             hidden_dim,
#             kernel_size=3,
#             padding=1,
#             groups=hidden_dim,   # depthwise
#             bias=True,
#         )

#         # self.act2 = nn.GELU()
#         self.act2 = nn.SiLU()
#         self.depthconv2 = nn.Conv2d(
#             hidden_dim,
#             hidden_dim,
#             kernel_size=3,
#             padding=1,
#             groups=hidden_dim,   # depthwise
#             bias=True,
#         )
#         self.act3 = nn.SiLU()

#         self.fc2 = nn.Linear(hidden_dim, embed_dim)
#         self.drop = nn.Dropout(mlp_p)

#         self.Hp = Hp
#         self.Wp = Wp

#     def forward(self, x):
#         # x: [B,HW,D]
#         B, HW, D = x.shape

#         x = self.fc1(x)
#         x = self.act1(x)

#         # reshape → conv → reshape back
#         x = x.transpose(1, 2).contiguous().view(B, -1, self.Hp, self.Wp)
#         res = x

#         x = self.dwconv(x)
#         x = self.act2(x)
#         x = self.depthconv2(x) + res  # depthwise residual
#         x = self.act3(x)

#         x = x.flatten(2).transpose(1, 2).contiguous()
#         x = self.fc2(x)

#         return self.drop(x)



# # =========================================================
# # Proxy FFT -> (radial x angular) bands
# # =========================================================
# class PatchProxySpectralBandsV2(nn.Module):
#     def __init__(
#         self,
#         embed_dim: int,
#         Hp: int,
#         Wp: int,
#         proxy_channels: int = 8,
#         radial_bins: int = 4,
#         angular_bins: int = 4,
#         eps: float = 1e-6,
#     ):
#         super().__init__()
#         assert radial_bins >= 1 and angular_bins >= 1
#         self.Hp, self.Wp = Hp, Wp
#         self.C = proxy_channels
#         self.R = radial_bins
#         self.A = angular_bins
#         self.K = radial_bins * angular_bins
#         self.eps = eps

#         self.to_proxy = nn.Linear(embed_dim, proxy_channels, bias=False)

#         # precompute band weights W: [K, Hp*Wp]
#         yy = torch.arange(Hp) - (Hp // 2)
#         xx = torch.arange(Wp) - (Wp // 2)
#         Y, X = torch.meshgrid(yy, xx, indexing="ij")
#         Y = Y.float()
#         X = X.float()

#         r = torch.sqrt(X * X + Y * Y)  # [Hp,Wp]
#         rmax = float(r.max().item()) if r.numel() else 1.0
#         redges = torch.linspace(0.0, rmax + 1e-6, steps=radial_bins + 1)

#         ang = torch.atan2(Y, X)
#         aedges = torch.linspace(-math.pi, math.pi + 1e-6, steps=angular_bins + 1)

#         band_id = torch.empty((Hp, Wp), dtype=torch.long)
#         k = 0
#         for ri in range(radial_bins):
#             rlo, rhi = redges[ri], redges[ri + 1]
#             rmask = (r >= rlo) & (r < rhi)
#             for ai in range(angular_bins):
#                 alo, ahi = aedges[ai], aedges[ai + 1]
#                 amask = (ang >= alo) & (ang < ahi)
#                 band_id[rmask & amask] = k
#                 k += 1

#         band_id = band_id.view(-1)
#         W = F.one_hot(band_id, num_classes=self.K).float().t()  # [K, Hp*Wp]
#         W = W / W.sum(dim=1, keepdim=True).clamp(min=1.0)
#         self.register_buffer("band_W", W, persistent=False)

#     def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         B, HW, D = patch_tokens.shape
#         proxy = self.to_proxy(patch_tokens)                         # [B,HW,C]
#         proxy = proxy.transpose(1, 2).contiguous()                  # [B,C,HW]
#         proxy = proxy.view(B, self.C, self.Hp, self.Wp).to(torch.float32)

#         Freq = torch.fft.fft2(proxy, norm="ortho")
#         mag = torch.abs(Freq).mean(dim=1)                           # [B,Hp,Wp]
#         mag = torch.fft.fftshift(mag, dim=(-2, -1))
#         mag = torch.log1p(mag + 1e-6)
#         mag = mag / (mag.mean(dim=(-2, -1), keepdim=True) + self.eps)

#         band_feat = mag.view(B, -1) @ self.band_W.t()               # [B,K]
#         return band_feat, mag


# # =========================================================
# # Band features -> band tokens
# # =========================================================
# class BandTokenEncoder(nn.Module):
#     def __init__(self, num_bands: int, embed_dim: int):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(1, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim),
#         )
#         self.embed_dim = embed_dim

#     def forward(self, band_feat: torch.Tensor) -> torch.Tensor:
#         B, K = band_feat.shape
#         x = band_feat.view(B * K, 1)
#         return self.proj(x).view(B, K, self.embed_dim)


# # =========================================================
# # Cross-attention
# # =========================================================
# class CrossAttention(nn.Module):
#     def __init__(self, embed_dim: int, num_head: int, attn_p: float = 0.0):
#         super().__init__()
#         assert embed_dim % num_head == 0
#         self.num_head = num_head
#         self.head_dim = embed_dim // num_head
#         self.scale = self.head_dim ** -0.5
#         self.attn_p = attn_p

#         self.q = nn.Linear(embed_dim, embed_dim, bias=True)
#         self.kv = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
#         self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

#     def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         B, N, D = x.shape
#         K = ctx.shape[1]

#         q = self.q(x).view(B, N, self.num_head, self.head_dim).transpose(1, 2)
#         kv = self.kv(ctx).view(B, K, 2, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         attn = (q * self.scale) @ k.transpose(-2, -1)
#         attn = attn.softmax(dim=-1)
#         attn = F.dropout(attn, p=(self.attn_p if self.training else 0.0))

#         out = attn @ v
#         out = out.transpose(1, 2).contiguous().view(B, N, D)
#         out = self.proj(out)
#         return out, attn.mean()


# # =========================================================
# # FiLM gate
# # =========================================================
# class FiLMGate(nn.Module):
#     def __init__(self, embed_dim: int, init_scale: float = 0.03):
#         super().__init__()
#         self.to_gb = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
#         nn.init.normal_(self.to_gb.weight, std=1e-4)
#         nn.init.zeros_(self.to_gb.bias)
#         self.init_scale = float(init_scale)

#     def forward(self, patch: torch.Tensor, cond: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
#         gamma, beta = self.to_gb(cond).chunk(2, dim=-1)
#         gamma = torch.tanh(gamma) * self.init_scale
#         beta  = torch.tanh(beta)  * self.init_scale
#         gamma = gamma.unsqueeze(1)
#         beta  = beta.unsqueeze(1)
#         return patch * (1.0 + alpha * gamma) + alpha * beta


# # =========================================================
# # Freq-conditioned self-attn
# # =========================================================
# class MultiHeadAttentionFreq(nn.Module):
#     def __init__(self, num_head: int, embed_dim: int, attn_p: float = 0.0, init_std: float = 1e-4):
#         super().__init__()
#         assert embed_dim % num_head == 0
#         self.num_head = num_head
#         self.head_dim = embed_dim // num_head
#         self.attn_p = attn_p

#         self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
#         self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

#         self.freq_to_q = nn.Linear(embed_dim, embed_dim, bias=True)
#         self.freq_to_k = nn.Linear(embed_dim, embed_dim, bias=True)
#         nn.init.normal_(self.freq_to_q.weight, std=init_std)
#         nn.init.normal_(self.freq_to_k.weight, std=init_std)
#         nn.init.zeros_(self.freq_to_q.bias)
#         nn.init.zeros_(self.freq_to_k.bias)

#     def forward(self, x: torch.Tensor, freq_token: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
#         B, N, D = x.shape
#         qkv = self.qkv(x).view(B, N, 3, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         f = freq_token[:, 0, :]
#         dq = self.freq_to_q(f).view(B, self.num_head, 1, self.head_dim)
#         dk = self.freq_to_k(f).view(B, self.num_head, 1, self.head_dim)

#         q = q + alpha * dq
#         k = k + alpha * dk

#         out = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.attn_p if self.training else 0.0))
#         out = out.transpose(1, 2).contiguous().view(B, N, D)
#         return self.proj(out)


# # =========================================================
# # Block
# # =========================================================
# class TransformerBlockFreqHeavy(nn.Module):
#     def __init__(self, embed_dim: int, num_head: int, mlp_ratio: float, attn_p: float, mlp_p: float, Hp: int, Wp: int):
#         super().__init__()
#         self.ln_gate = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.film = FiLMGate(embed_dim, init_scale=0.03)

#         self.ln_xattn = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.xattn = CrossAttention(embed_dim, num_head, attn_p=attn_p)

#         self.ln_attn = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.attn = MultiHeadAttentionFreq(num_head, embed_dim, attn_p=attn_p)

#         self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.mlp = ConvMLP(embed_dim, int(embed_dim * mlp_ratio), Hp, Wp, mlp_p)

#     def forward(
#         self,
#         x: torch.Tensor,
#         band_tokens: torch.Tensor,
#         freq_token_for_cond: torch.Tensor,
#         freq_summary: torch.Tensor,
#         alpha: torch.Tensor,
#     ):
#         # split
#         cls_freq = x[:, :2, :]      # [B,2,D]
#         patch    = x[:, 2:, :]      # [B,HW,D]

#         # gate patches
#         patch = self.film(self.ln_gate(patch), freq_summary, alpha=alpha)

#         # cross-attn on patches
#         xa, xattn_mean = self.xattn(self.ln_xattn(patch), band_tokens)
#         patch = patch + xa

#         # re-join tokens for self-attn
#         x = torch.cat([cls_freq, patch], dim=1)

#         # self-attn over full sequence
#         x = x + self.attn(self.ln_attn(x), freq_token=freq_token_for_cond, alpha=alpha)

#         # MLP ONLY on patches (this is the critical fix)
#         cls_freq = x[:, :2, :]
#         patch    = x[:, 2:, :]
#         patch    = patch + self.mlp(self.ln_mlp(patch))
#         x = torch.cat([cls_freq, patch], dim=1)

#         return x, xattn_mean



# # =========================================================
# # Main model (FX-safe)
# # =========================================================
# class VisionTransformerFreqV3(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         num_classes: int,
#         image_size: int = 224,
#         patch_size: int = 16,
#         embed_dim: int = 768,
#         depth: int = 12,
#         num_head: int = 12,
#         mlp_ratio: float = 4.0,
#         attn_p: float = 0.0,
#         mlp_p: float = 0.0,
#         pos_p: float = 0.0,
#         proxy_channels: int = 8,
#         radial_bins: int = 4,
#         angular_bins: int = 4,
#         detach_freq_token_update: bool = True,
#         alpha: float = 0.10,
#         anchor_weight: float = 0.05,
#         usage_reg_weight: float = 0.01,
#         usage_target: float = 0.15,
#     ):
#         super().__init__()
#         assert image_size % patch_size == 0

#         self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)
#         self.Hp = image_size // patch_size
#         self.Wp = image_size // patch_size
#         self.num_tokens = self.patch_embed.num_patches + 2

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.freq_token_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
#         self.pos_drop = nn.Dropout(pos_p)

#         self.spec = PatchProxySpectralBandsV2(
#             embed_dim=embed_dim,
#             Hp=self.Hp,
#             Wp=self.Wp,
#             proxy_channels=proxy_channels,
#             radial_bins=radial_bins,
#             angular_bins=angular_bins,
#         )
#         self.K = self.spec.K
#         self.band_enc = BandTokenEncoder(num_bands=self.K, embed_dim=embed_dim)

#         self.freq_summary_enc = nn.Sequential(
#             nn.LayerNorm(self.K),
#             nn.Linear(self.K, embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim),
#         )

#         self.blocks = nn.ModuleList([
#             TransformerBlockFreqHeavy(embed_dim, num_head, mlp_ratio, attn_p, mlp_p, self.Hp, self.Wp)
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.head = nn.Linear(embed_dim, num_classes)

#         self.detach_freq_token_update = bool(detach_freq_token_update)

#         # Buffers (FX-safe, no img.device/img.dtype usage)
#         self.register_buffer("alpha_buf", torch.tensor(float(alpha)), persistent=True)
#         self.register_buffer("usage_target_buf", torch.tensor(float(usage_target)), persistent=True)

#         self.anchor_weight = float(anchor_weight)
#         self.usage_reg_weight = float(usage_reg_weight)

#         self._init_params()

#     def _init_params(self):
#         trunc_normal_(self.pos_embed, std=0.02)
#         trunc_normal_(self.cls_token, std=0.02)
#         trunc_normal_(self.freq_token_bias, std=0.02)

#     def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         B = img.size(0)

#         patch = self.patch_embed(img)  # [B,HW,D]

#         # FX-safe alpha match: create a scalar with same dtype/device as patch using tensor ops
#         alpha = self.alpha_buf * patch.new_ones(())  # [] scalar, same dtype/device as patch
#         usage_target = self.usage_target_buf * patch.new_ones(())  # [] scalar

#         band_feat, mag_grid = self.spec(patch)          # [B,K], [B,Hp,Wp]
#         band_tokens = self.band_enc(band_feat)          # [B,K,D]
#         freq_summary = self.freq_summary_enc(band_feat) # [B,D]
#         freq_summary = F.layer_norm(freq_summary, (freq_summary.shape[-1],))

#         freq_tok0 = (freq_summary.unsqueeze(1) + self.freq_token_bias).to(patch.dtype)  # [B,1,D]

#         cls = self.cls_token.expand(B, -1, -1).to(patch.dtype)
#         x = torch.cat([cls, freq_tok0, patch], dim=1)  # [B,2+HW,D]
#         x = x + self.pos_embed.to(x.dtype)
#         x = self.pos_drop(x)

#         xattn_means = []
#         for blk in self.blocks:
#             freq_token_for_cond = x[:, 1:2, :]
#             if self.detach_freq_token_update:
#                 freq_token_for_cond = freq_token_for_cond.detach()

#             x, xattn_mean = blk(
#                 x,
#                 band_tokens=band_tokens.to(x.dtype),
#                 freq_token_for_cond=freq_token_for_cond,
#                 freq_summary=freq_summary.to(x.dtype),
#                 alpha=alpha,
#             )
#             xattn_means.append(xattn_mean)

#         x = self.norm(x)
#         logits = self.head(x[:, 0, :])

#         aux: Dict[str, torch.Tensor] = {
#             "freq_token": x[:, 1:2, :],
#             "patch_token": x[:, 2:, :]
#         }

#         return logits, aux



import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# =========================================================
# Utils
# =========================================================
def trunc_normal_(tensor, mean=0.0, std=0.02):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std).add_(mean)
    return tensor


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [..., D]
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


# =========================================================
# Patch embedding
# =========================================================
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, image_size=224, embed_dim=768):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                               # [B,D,Hp,Wp]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B,HW,D]
        return x


# =========================================================
# MLP (simplified for quant robustness)
#   fc1 -> GELU -> DWConv(+res) -> SiLU -> fc2
# =========================================================
class ConvMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, Hp, Wp, mlp_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act1 = nn.GELU()

        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim ,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,   # depthwise
            bias=True,
        )
        self.act2 = nn.SiLU()

        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(mlp_p)

        self.Hp = Hp
        self.Wp = Wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,HW,D]
        B, HW, D = x.shape
        x = self.fc1(x)
        x = self.act1(x)

        x = x.transpose(1, 2).contiguous().view(B, -1, self.Hp, self.Wp)
        x = self.dwconv(x)
        x = self.act2(x)
 
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.fc2(x)
        return self.drop(x)


# =========================================================
# Proxy FFT bands from PATCH TOKENS (your original idea)
#   - add option to detach tokens for stability
# =========================================================
class PatchProxySpectralBandsV2(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        Hp: int,
        Wp: int,
        proxy_channels: int = 8,
        radial_bins: int = 4,
        angular_bins: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert radial_bins >= 1 and angular_bins >= 1
        self.Hp, self.Wp = Hp, Wp
        self.C = proxy_channels
        self.R = radial_bins
        self.A = angular_bins
        self.K = radial_bins * angular_bins
        self.eps = eps

        self.to_proxy = nn.Linear(embed_dim, proxy_channels, bias=False)

        # Precompute band weights W: [K, Hp*Wp] (FX-safe)
        yy = torch.arange(Hp) - (Hp // 2)
        xx = torch.arange(Wp) - (Wp // 2)
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        Y = Y.float()
        X = X.float()

        r = torch.sqrt(X * X + Y * Y)  # [Hp,Wp]
        rmax = float(r.max().item()) if r.numel() else 1.0
        redges = torch.linspace(0.0, rmax + 1e-6, steps=radial_bins + 1)

        ang = torch.atan2(Y, X)  # [-pi,pi]
        aedges = torch.linspace(-math.pi, math.pi + 1e-6, steps=angular_bins + 1)

        band_id = torch.zeros((Hp, Wp), dtype=torch.long)
        k = 0
        for ri in range(radial_bins):
            rlo, rhi = redges[ri], redges[ri + 1]
            rmask = (r >= rlo) & (r < rhi)
            for ai in range(angular_bins):
                alo, ahi = aedges[ai], aedges[ai + 1]
                amask = (ang >= alo) & (ang < ahi)
                band_id[rmask & amask] = k
                k += 1

        band_id = band_id.view(-1)
        W = F.one_hot(band_id, num_classes=self.K).float().t()  # [K, Hp*Wp]
        W = W / W.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer("band_W", W, persistent=False)

    def forward(self, patch_tokens: torch.Tensor, detach_tokens: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # patch_tokens: [B,HW,D]
        if detach_tokens:
            patch_tokens = patch_tokens.detach()

        B, HW, D = patch_tokens.shape
        proxy = self.to_proxy(patch_tokens)                         # [B,HW,C]
        proxy = proxy.transpose(1, 2).contiguous()                  # [B,C,HW]
        proxy = proxy.view(B, self.C, self.Hp, self.Wp).to(torch.float32)

        Freq = torch.fft.fft2(proxy, norm="ortho")
        mag = torch.abs(Freq).mean(dim=1)                           # [B,Hp,Wp]
        mag = torch.fft.fftshift(mag, dim=(-2, -1))
        mag = torch.log1p(mag + 1e-6)
        mag = mag / (mag.mean(dim=(-2, -1), keepdim=True) + self.eps)

        band_feat = mag.view(B, -1) @ self.band_W.t()               # [B,K]
        return band_feat, mag


# =========================================================
# Band features -> band tokens
# =========================================================
class BandTokenEncoder(nn.Module):
    def __init__(self, num_bands: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, band_feat: torch.Tensor) -> torch.Tensor:
        B, K = band_feat.shape
        x = band_feat.view(B * K, 1)
        return self.proj(x).view(B, K, self.embed_dim)


# =========================================================
# Cross-attention (stabilized: RMSNorm ctx)
# =========================================================
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

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,N,D], ctx: [B,K,D]
        B, N, D = x.shape
        K = ctx.shape[1]

        # normalize ctx for stability (esp under quant)
        ctxn = rms_norm(ctx)

        q = self.q(x).view(B, N, self.num_head, self.head_dim).transpose(1, 2)
        kv = self.kv(ctxn).view(B, K, 2, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # stable logits path (explicit softmax)
        attn_logits = (q * self.scale) @ k.transpose(-2, -1)
        attn_logits = attn_logits.clamp(-10, 10)
        attn = attn_logits.softmax(dim=-1)
        attn = F.dropout(attn, p=(self.attn_p if self.training else 0.0))

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        return out, attn.mean()


# =========================================================
# FiLM gate (patch gating with freq_summary)
# =========================================================
class FiLMGate(nn.Module):
    def __init__(self, embed_dim: int, init_scale: float = 0.03):
        super().__init__()
        self.to_gb = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        nn.init.normal_(self.to_gb.weight, std=1e-4)
        nn.init.zeros_(self.to_gb.bias)
        self.init_scale = float(init_scale)

    def forward(self, patch: torch.Tensor, cond: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # patch: [B,HW,D], cond: [B,D], alpha: scalar []
        gamma, beta = self.to_gb(cond).chunk(2, dim=-1)  # [B,D]
        gamma = torch.tanh(gamma) * self.init_scale
        beta  = torch.tanh(beta)  * self.init_scale
        gamma = gamma.unsqueeze(1)
        beta  = beta.unsqueeze(1)
        return patch * (1.0 + alpha * gamma) + alpha * beta


# =========================================================
# NEW: Spectral-Similarity Self-Attn (quant-robust)
#   - RMSNorm Q/K
#   - Clamp logits
#   - Add spectral similarity bias to logits (band-feat)
#   - Head-wise spectral gating (multiplicative), safer than additive dq/dk
# =========================================================
class SpectralSimilarityAttention(nn.Module):
    def __init__(
        self,
        num_head: int,
        embed_dim: int,
        band_dim: int,
        attn_p: float = 0.0,
        logit_clip: float = 10.0,
        spectral_bias_init: float = 0.02,
        head_gate_init: float = 0.05,
    ):
        super().__init__()
        assert embed_dim % num_head == 0
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scale = self.head_dim ** -0.5
        self.attn_p = attn_p
        self.logit_clip = float(logit_clip)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # spectral bias strength (learnable, starts small)
        self.spec_bias_scale = nn.Parameter(torch.tensor(float(spectral_bias_init)))

        # per-head spectral gates from freq token (multiplicative on head outputs)
        self.freq_to_head_gate = nn.Linear(embed_dim, num_head, bias=True)
        nn.init.zeros_(self.freq_to_head_gate.bias)
        nn.init.normal_(self.freq_to_head_gate.weight, std=1e-4)
        self.head_gate_init = float(head_gate_init)

        # project token band-features to a small spectral descriptor (per token)
        # band_feat_tok: [B,N,band_dim] -> spectral descriptor [B,N,sd]
        sd = min(16, max(4, band_dim))
        self.band_tok_proj = nn.Linear(band_dim, sd, bias=False)

    def forward(
        self,
        x: torch.Tensor,                 # [B,N,D]
        freq_token: torch.Tensor,        # [B,1,D]
        band_feat_tok: torch.Tensor,     # [B,N,band_dim]  (per-token spectral/band descriptor)
        alpha: torch.Tensor,             # scalar []
    ) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).view(B, N, 3, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,N,hd]

        # quant-robust normalization for similarity
        q = rms_norm(q)
        k = rms_norm(k)

        # base logits
        logits = (q * self.scale) @ k.transpose(-2, -1)  # [B,H,N,N]

        # spectral similarity bias (token-token)
        # Make a compact spectral descriptor per token; use cosine-ish similarity
        s = self.band_tok_proj(band_feat_tok.to(x.dtype))  # [B,N,sd]
        s = rms_norm(s)
        ssim = s @ s.transpose(-2, -1)  # [B,N,N]
        logits = logits + (alpha * self.spec_bias_scale) * ssim.unsqueeze(1)

        # clamp for stability
        logits = logits.clamp(-self.logit_clip, self.logit_clip)

        attn = logits.softmax(dim=-1)
        attn = F.dropout(attn, p=(self.attn_p if self.training else 0.0))

        out = attn @ v  # [B,H,N,hd]

        # head gate from freq token (multiplicative, safer than additive q/k shifts)
        f = freq_token[:, 0, :]  # [B,D]
        g = torch.tanh(self.freq_to_head_gate(f)) * self.head_gate_init  # [B,H]
        out = out * (1.0 + alpha * g[:, :, None, None])

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.proj(out)


# =========================================================
# Token band features (cheap, derived from your global band_feat + per-token proxy)
#   You want something per-token so attention can use spectral bias.
#   This uses the same proxy linear projection but WITHOUT FFT per-token (too expensive).
#   We approximate per-token band features by mixing global band_feat with token-local proxy energy.
# =========================================================
class TokenBandFeatures(nn.Module):
    def __init__(self, embed_dim: int, band_dim: int):
        super().__init__()
        # local token descriptor
        self.local = nn.Sequential(
            nn.Linear(embed_dim, band_dim),
            nn.GELU(),
            nn.Linear(band_dim, band_dim),
        )
        # global -> token modulation
        self.global_to_gate = nn.Linear(band_dim, band_dim, bias=True)
        nn.init.zeros_(self.global_to_gate.bias)
        nn.init.normal_(self.global_to_gate.weight, std=1e-4)

    def forward(self, x_tokens: torch.Tensor, global_band: torch.Tensor) -> torch.Tensor:
        # x_tokens: [B,N,D], global_band: [B,K] where K == band_dim here (we’ll set K=band_dim)
        # If your K != band_dim, just project global_band before passing here.
        local = self.local(x_tokens)  # [B,N,band_dim]
        gate = torch.sigmoid(self.global_to_gate(global_band)).unsqueeze(1)  # [B,1,band_dim]
        return local * gate


# =========================================================
# Block (freq-heavy but stable)
# =========================================================
class TransformerBlockFreqHeavy(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_head: int,
        mlp_ratio: float,
        attn_p: float,
        mlp_p: float,
        Hp: int,
        Wp: int,
        band_dim_for_attn: int,
    ):
        super().__init__()
        self.ln_gate = nn.LayerNorm(embed_dim, eps=1e-6)
        self.film = FiLMGate(embed_dim, init_scale=0.5)

        # self.ln_xattn = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.xattn = CrossAttention(embed_dim, num_head, attn_p=attn_p)

        self.ln_attn = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = SpectralSimilarityAttention(
            num_head=num_head,
            embed_dim=embed_dim,
            band_dim=band_dim_for_attn,
            attn_p=attn_p,
            logit_clip=10.0,
        )

        self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = ConvMLP(embed_dim, int(embed_dim * mlp_ratio), Hp, Wp, mlp_p)

    def forward(
        self,
        x: torch.Tensor,                 # [B,2+HW,D]
        band_tokens: torch.Tensor,       # [B,K,D]
        freq_token_for_cond: torch.Tensor,  # [B,1,D]
        freq_summary: torch.Tensor,      # [B,D]
        token_band_feat: torch.Tensor,   # [B,2+HW,band_dim]
        alpha: torch.Tensor,             # scalar []
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # split
        cls_freq = x[:, :2, :]      # [B,2,D]
        patch    = x[:, 2:, :]      # [B,HW,D]

        # gate patches (freq_summary)
        patch = self.film(self.ln_gate(patch), freq_summary, alpha=alpha)

        # cross-attn on patches (ctx=band tokens)
        # xa, xattn_mean = self.xattn(self.ln_xattn(patch), band_tokens)
        # patch = patch + xa

        # re-join tokens for self-attn
        x = torch.cat([cls_freq, patch], dim=1)

        # spectral-similarity self-attn over full sequence
        x = x + self.attn(self.ln_attn(x), freq_token=freq_token_for_cond, band_feat_tok=token_band_feat, alpha=alpha)

        # MLP ONLY on patches
        cls_freq = x[:, :2, :]
        patch    = x[:, 2:, :]
        patch    = patch + self.mlp(self.ln_mlp(patch))
        x = torch.cat([cls_freq, patch], dim=1)

        # return x, xattn_mean
        return x, torch.tensor(0.0, device='cuda')  # placeholder for xattn_mean


# =========================================================
# Main model (optimized + stable under 2-bit ACT)
#   Key changes vs your V3:
#   - Self-attn uses spectral similarity bias + head gating (no additive dq/dk)
#   - Q/K RMSNorm + logit clamp
#   - Optional detach of spectral extraction from patch tokens (warmup stability)
#   - Token-band features provided to attention (cheap)
# =========================================================
class VisionTransformerFreqV4(nn.Module):
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
        proxy_channels: int = 8,
        radial_bins: int = 4,
        angular_bins: int = 4,
        detach_freq_token_update: bool = True,
        detach_spec_from_patch: bool = True,  # <-- IMPORTANT for 2-bit convergence
        alpha: float = 0.10,
    ):
        super().__init__()
        assert image_size % patch_size == 0

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(in_channels, patch_size, image_size, embed_dim)
        self.Hp = image_size // patch_size
        self.Wp = image_size // patch_size
        self.num_tokens = self.patch_embed.num_patches + 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.freq_token_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(pos_p)

        self.spec = PatchProxySpectralBandsV2(
            embed_dim=embed_dim,
            Hp=self.Hp,
            Wp=self.Wp,
            proxy_channels=proxy_channels,
            radial_bins=radial_bins,
            angular_bins=angular_bins,
        )
        self.K = self.spec.K
        self.band_enc = BandTokenEncoder(num_bands=self.K, embed_dim=embed_dim)

        # Global band -> freq_summary
        self.freq_summary_enc = nn.Sequential(
            nn.LayerNorm(self.K),
            nn.Linear(self.K, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # For attention spectral bias, we want a modest band_dim.
        # Easiest: use K directly (works fine).
        self.band_dim_for_attn = self.K

        # Provide per-token band features (cheap approximation)
        self.token_band = TokenBandFeatures(embed_dim=embed_dim, band_dim=self.band_dim_for_attn)

        self.blocks = nn.ModuleList([
            TransformerBlockFreqHeavy(
                embed_dim=embed_dim,
                num_head=num_head,
                mlp_ratio=mlp_ratio,
                attn_p=attn_p,
                mlp_p=mlp_p,
                Hp=self.Hp,
                Wp=self.Wp,
                band_dim_for_attn=self.band_dim_for_attn,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self.detach_freq_token_update = bool(detach_freq_token_update)
        self.detach_spec_from_patch = bool(detach_spec_from_patch)

        self.register_buffer("alpha_buf", torch.tensor(float(alpha)), persistent=True)

        self._init_params()

    def _init_params(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.freq_token_bias, std=0.02)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = img.size(0)

        patch = self.patch_embed(img)  # [B,HW,D]

        # FX-safe alpha scalar matching dtype/device
        alpha = self.alpha_buf * patch.new_ones(())

        # Global spectral bands from patch tokens (optionally detached for stability under 2-bit)
        band_feat, mag_grid = self.spec(patch, detach_tokens=self.detach_spec_from_patch)  # [B,K]
        band_tokens = self.band_enc(band_feat)  # [B,K,D]

        freq_summary = self.freq_summary_enc(band_feat)  # [B,D]
        freq_summary = F.layer_norm(freq_summary, (freq_summary.shape[-1],))

        # Construct initial freq token from summary + bias
        freq_tok0 = (freq_summary.unsqueeze(1) + self.freq_token_bias).to(patch.dtype)  # [B,1,D]

        # tokens: [CLS, FREQ, PATCH...]
        cls = self.cls_token.expand(B, -1, -1).to(patch.dtype)
        x = torch.cat([cls, freq_tok0, patch], dim=1)  # [B,2+HW,D]
        x = x + self.pos_embed.to(x.dtype)
        x = self.pos_drop(x)

        # build per-token band features for attention bias
        # token_band expects global_band dim == K
        token_band_feat = self.token_band(x, band_feat.to(x.dtype))  # [B,2+HW,K]

        xattn_means = []
        for blk in self.blocks:
            freq_token_for_cond = x[:, 1:2, :]
            if self.detach_freq_token_update:
                freq_token_for_cond = freq_token_for_cond.detach()

            x, xattn_mean = blk(
                x,
                band_tokens=band_tokens.to(x.dtype),
                freq_token_for_cond=freq_token_for_cond,
                freq_summary=freq_summary.to(x.dtype),
                token_band_feat=token_band_feat.to(x.dtype),
                alpha=alpha,
            )
            xattn_means.append(xattn_mean)

        x = self.norm(x)
        logits = self.head(x[:, 0, :])

        aux: Dict[str, torch.Tensor] = {
            "freq_token": x[:, 1:2, :],
            "patch_token": x[:, 2:, :].transpose(1, 2).contiguous().view(B, self.embed_dim, self.Hp, self.Wp),
            "xattn_mean": torch.stack(xattn_means).mean(),
        }
        return logits, aux
