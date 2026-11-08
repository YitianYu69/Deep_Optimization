import torch 
from torch import nn
from torch.nn import functional as F

from .functional import do_linear, _prepare_fp8_plan
from ._cached_weights import CachedWeightModule


class DOMultiheadAttention(CachedWeightModule):
    def __init__(self, original: nn.MultiheadAttention, *, plan: dict):
        super().__init__()

        if type(original) is not nn.MultiheadAttention:
            raise TypeError("original musy be a standard nn.MultiheadAttention")
        if original.bias_k is not None or original.bias_v is not None:
            raise NotImplementedError("Learned bias key/value tokens are not supported yet")
        if original.add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported yet")
        
        self.plan = _prepare_fp8_plan(plan)
        self._projection_scaling = nn.ModuleDict()
        if self.plan["fp8_training"]["scaling_recipe"] == "delayed":
            from ..scaling_strategy.delayed import DelayedScalingState

            device = next(original.parameters()).device
            for role in ("qkv", "q", "k", "v", "out"):
                self._projection_scaling[role] = DelayedScalingState(
                    self.plan["fp8_training"]["delayed_scaling"], self.plan["fp8_training"]["fp8_dtypes"], device=device)

        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.kdim = original.kdim
        self.vdim = original.vdim
        self.dropout = original.dropout
        self.batch_first = original.batch_first
        self.add_zero_attn = original.add_zero_attn
        self._qkv_same_embed_dim = original._qkv_same_embed_dim

        # Preserve registration order as well as identity: optimizer state
        # dictionaries associate saved slots with parameter-list positions.
        for name, parameter in original._parameters.items():
            self.register_parameter(name, parameter)
        
        self.out_proj = original.out_proj
        self.training = original.training
    
    @classmethod
    def from_mha(cls, original: nn.MultiheadAttention, *, plan: dict) -> "DOMultiheadAttention":
        return cls(original, plan=plan)

    def _project(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        *,
        self_attention: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep projection execution behind do_linear so future activation
        # codecs/sparse backends do not change attention layout or mask logic.
        if self.in_proj_weight is not None:
            if self_attention:
                packed = do_linear(
                    query,
                    self.in_proj_weight, self.in_proj_bias,
                    plan=self.plan, _weight_cache=self._fp8_weight_cache,
                    _scaling_state=self._projection_scaling["qkv"] if self._projection_scaling else None,
                    _state_training=self.training,
                )
                return packed.chunk(3, dim=-1)
            weights = self.in_proj_weight.chunk(3, dim=0)
        else:
            weights = (
                self.q_proj_weight,
                self.k_proj_weight,
                self.v_proj_weight,
            )
        
        if self.in_proj_bias is None:
            biases = (None, None, None)
        else:
            biases = self.in_proj_bias.chunk(3, dim=0)
        
        q = do_linear(
            query, weights[0], biases[0], plan=self.plan, _weight_cache=self._fp8_weight_cache,
            _scaling_state=self._projection_scaling["q"] if self._projection_scaling else None, _state_training=self.training,
        )
        k = do_linear(
            key, weights[1], biases[1], plan=self.plan, _weight_cache=self._fp8_weight_cache,
            _scaling_state=self._projection_scaling["k"] if self._projection_scaling else None, _state_training=self.training,
        )
        v = do_linear(
            value, weights[2], biases[2], plan=self.plan, _weight_cache=self._fp8_weight_cache,
            _scaling_state=self._projection_scaling["v"] if self._projection_scaling else None, _state_training=self.training,
        )
        return q, k, v

    @staticmethod
    def _additive_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Attention masks must be tensors")
        if mask.is_nested or mask.layout != torch.strided:
            raise TypeError("Attention masks must be dense tensors")
        if mask.device != reference.device:
            raise ValueError("Attention masks must be on the input device")
        if mask.dtype == torch.bool:
            additive = torch.zeros(
                mask.shape,
                device=mask.device,
                dtype=torch.float32
            )
            return additive.masked_fill(mask, float("-inf"))

        if mask.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError("Attention masks must be boolean, FP32, BF16, or FP16")
        # Keep finite biases finite and combine masks before any low-precision
        # conversion. SDPA accepts FP32 additive masks with FP16/BF16 queries.
        return mask.float()

    def _attention_mask(
        self,
        attn_mask,
        key_padding_mask,
        reference,
        *,
        batch_size,
        target_length,
        source_length,
        unbatched
    ):
        combined = None

        if attn_mask is not None:
            mask = self._additive_mask(attn_mask, reference)

            if tuple(mask.shape) == (target_length, source_length):
                combined = mask.reshape(1, 1, target_length, source_length)
            elif tuple(mask.shape) == (batch_size * self.num_heads, target_length, source_length):
                combined = mask.reshape(batch_size, self.num_heads, target_length, source_length)
            else:
                raise ValueError(
                    "attn_mask must have shape [L, S] "
                    "or [batch_size * num_heads, L, S]"
                )
        if key_padding_mask is not None:
            padding = self._additive_mask(key_padding_mask, reference)

            expected_shape = (source_length,) if unbatched else (batch_size, source_length)
            if tuple(padding.shape) != expected_shape:
                raise ValueError(
                    f"Key_padding_mask must have shape {expected_shape}"
                )
            
            padding = padding.reshape(batch_size, 1, 1, source_length)
            combined = padding if combined is None else combined + padding
        return combined

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for flag in (need_weights, average_attn_weights, is_causal):
            if type(flag) is not bool:
                raise TypeError("Attention flags must be booleans")
        
        for tensor in(query, key, value):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Query, key, and value must be tensors")
            if tensor.is_nested or tensor.layout != torch.strided:
                raise NotImplementedError("Only dense query, key and value tensors are supported")
            if tensor.ndim not in (2, 3):
                raise ValueError("Query, key, and value must be 2D or 3D")

        if not (query.ndim == key.ndim == value.ndim):
            raise ValueError("Query, key, and value must have the same rank")
        if not (query.device == key.device == value.device):
            raise ValueError("Query, key, and value must share a device")
        if not (query.dtype == key.dtype == value.dtype):
            raise TypeError("Query, key, and value must share a dtype")
        if is_causal and attn_mask is None:
            raise ValueError(
                "This adapter requires an explicit attn_mask "
                "when is_causal=True"
            )

        self_attention = query is key and key is value
        unbatched = query.ndim == 2

        if unbatched:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        elif not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, target_length, _ = query.shape
        source_length = key.shape[1]

        if key.shape[0] != batch_size or value.shape[0] != batch_size:
            raise ValueError("Query, key, and value batch sizes differ")
        if value.shape[1] != source_length:
            raise ValueError("Key and value sequence lengths differ")
        for tensor, expected_width in zip(
            (query, key, value),
            (self.embed_dim, self.kdim, self.vdim),
        ):
            if tensor.shape[-1] != expected_width:
                raise ValueError("Query, key, or value has an incorrect feature dimension")
        
        q, k, v = self._project(query, key, value, self_attention=self_attention)
        q = q.reshape(batch_size, target_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, source_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, source_length, self.num_heads, self.head_dim).transpose(1, 2)

        mask = self._attention_mask(
            attn_mask, key_padding_mask,
            q,
            batch_size=batch_size,
            target_length=target_length,
            source_length=source_length,
            unbatched=unbatched
        )
        dropout_p = self.dropout if self.training else 0.0

        with torch.autocast(device_type=q.device.type, enabled=False):
            if need_weights:
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (self.head_dim ** -0.5)

                if mask is not None:
                    scores = scores + mask.float()
                
                probabilities = torch.softmax(scores, dim=-1)
                probabilities = F.dropout(probabilities, p=dropout_p, training=self.training)

                attended = torch.matmul(probabilities, v.float()).to(dtype=q.dtype)
                weights = probabilities.to(dtype=q.dtype)
            else:
                attended = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
                weights = None
            
        attended = attended.transpose(1, 2).reshape(batch_size, target_length, self.embed_dim)
        output = do_linear(
            attended, self.out_proj.weight, self.out_proj.bias,
            plan=self.plan, _weight_cache=self._fp8_weight_cache,
            _scaling_state=self._projection_scaling["out"] if self._projection_scaling else None, _state_training=self.training,
        )

        if weights is not None and average_attn_weights:
            weights = weights.mean(dim=1)
        if unbatched:
            output = output.squeeze(0)
            if weights is not None:
                weights = weights.squeeze(0)
        elif not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, weights

    def extra_repr(self) -> str:
        backend = self.plan["fp8_training"]["backend"]
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"batch_first={self.batch_first}, "
            f"dropout={self.dropout}, "
            f"backend={backend!r}"
        )
