"""Native SDPA VJP with explicit QKV, softmax, RNG and kernel-state results."""

from dataclasses import replace
from math import sqrt

import torch
from torch.nn.attention import SDPBackend

from ..ir import AuxiliaryTensorMeta, CallNode, TensorValue
from .backward_rules import GradWrapper
from .semantic import result


# Private execution variants are saved separately from PyTorch's backend IDs.
_FLASH_ATTN_2 = 10
_CPU_FLASH_ATTENTION = 11


def _flash_operand(value):
    """Match public CUDA SDPA's last-dimension padding for private kernels."""
    padding = -value.shape[-1] % 8
    return torch.nn.functional.pad(value, (0, padding)) if padding else value


class SDPA:
    op = "autodiff_sdpa"
    names = ("q_data", "q_step", "q_min", "k_data", "k_step", "k_min", "v_data", "v_step", "v_min",
             "lse", "probability", "drop_probability", "drop_mask", "seed", "offset", "cu_q", "cu_k", "implementation")

    def __init__(self, node, context):
        self.options = node.options
        self.metas = tuple(value.meta if value is not None else None for value in node.args)
        self.output_dtype = node.value.meta.dtype
        self.needs = tuple(metadata is not None and metadata.requires_grad for metadata in self.metas)
        self.compression = self.options["compression"]
        from deepopt._attention_policy import effective_attention_bits

        # The internal SDPA ABI historically uses None for disabled and a
        # codec descriptor for enabled. Preserve that ABI while respecting
        # an explicit enabled=False in a complete configuration.
        self.bits = effective_attention_bits({"enabled": True, **self.compression}) if self.compression is not None else 0
        self.scale = float(self.metas[0].shape[-1] ** -0.5 if self.options["scale"] is None else self.options["scale"])

    def lower(self, node):
        states = tuple(TensorValue(f"{node.value.name}__{name}", meta)
                       for name, meta in zip(self.names, self.state_metadata()))
        return CallNode(self.op, self.forward, node.args, {}, (node.value, *states), self)

    def state_metadata(self):
        """The saved ABI is typed; backend-owned dimensions remain unknown.

        Fused kernels may return undefined cumulative lengths, or RNG state
        when dropout is disabled. Keep their slots and their actual absence;
        an empty descriptor is a different, present value.
        """
        metadata = []
        for name in self.names:
            if name == "implementation":
                meta = AuxiliaryTensorMeta((), torch.int64, "cpu")
            elif name in ("cu_q", "cu_k"):
                meta = AuxiliaryTensorMeta(dtype=torch.int32, device=self.metas[0].device, presence="optional")
            else:
                meta = AuxiliaryTensorMeta(presence="optional" if name in ("seed", "offset") else "required")
            metadata.append(meta)
        return tuple(metadata)

    def _validate_saved(self, implementation, saved):
        """Validate the selected kernel's state before crossing the bridge."""
        if len(saved) != len(self.names):
            raise ValueError("Native SDPA returned an incorrectly sized saved-state tuple")
        for name, value, meta in zip(self.names, saved, self.state_metadata()):
            optional = meta.presence == "optional"
            if name in ("seed", "offset") and self.options["dropout_p"]:
                # CPU/math dropout saves its exact dropout mask instead.
                optional = implementation in (int(SDPBackend.MATH), _CPU_FLASH_ATTENTION)
                if name == "offset" and implementation == _FLASH_ATTN_2:
                    optional = True  # The external adapter owns one RNG state tensor.
            label = f"Native SDPA implementation {implementation} state {name!r}"
            if value is None and optional:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{label} must be {'a tensor or None' if optional else 'a tensor'}")
            if meta.shape is not None and tuple(value.shape) != meta.shape:
                raise ValueError(f"{label} has an invalid shape")
            if meta.dtype is not None and value.dtype != meta.dtype:
                raise ValueError(f"{label} has an invalid dtype")
            if name in ("cu_q", "cu_k") and value.ndim != 1:
                raise ValueError(f"{label} must be a cumulative-length vector")
            if meta.device is not None:
                device = torch.device(meta.device)
                if value.device.type != device.type or (device.index is not None and value.device.index != device.index):
                    raise ValueError(f"{label} has an invalid device")

    def inference(self, q, k, v, mask):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, mask,
            self.options["dropout_p"], self.options["is_causal"], scale=self.options["scale"],
            enable_gqa=self.options["enable_gqa"])

    def _pack(self, value, empty):
        if not self.bits or not value.numel():
            return value, empty, empty
        from deepopt.opt.quantization.activation import _quantize_pack_strided_activation

        packed = _quantize_pack_strided_activation(value, self.compression["group_size"], bits=self.bits)
        return packed.data, packed.step, packed.minimum

    def _restore(self, values, shape, dtype):
        if not self.bits:
            return values[0]
        from math import prod
        from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation

        if not prod(shape):
            return values[0].reshape(shape)
        packed = PackedActivation(*values, (prod(shape[:-1]), shape[-1]), self.compression["group_size"], bits=self.bits)
        return dequantize_unpack_activation(packed, dtype=dtype).reshape(shape)

    def _math_shapes(self):
        """Match the operands saved by public math SDPA's two batched GEMMs."""
        q, k, v = (metadata.shape for metadata in self.metas[:3])
        if self.options["enable_gqa"]:
            k, v = (k[0], q[1], *k[2:]), (v[0], q[1], *v[2:])
        qk_batch = torch.broadcast_shapes(q[:-2], k[:-2])
        score = (*qk_batch, q[-2], k[-2])
        if self.metas[3] is not None:
            score = torch.broadcast_shapes(score, self.metas[3].shape)
        value_batch = torch.broadcast_shapes(score[:-2], v[:-2])
        return ((*qk_batch, *q[-2:]), (*qk_batch, k[-1], k[-2]), (*value_batch, *v[-2:]))

    def forward(self, q, k, v, mask):
        with torch.autocast(q.device.type, enabled=False):
            return self._forward(q, k, v, mask)

    def _flash_forward(self, q, k, v, p, causal):
        values = torch.ops.aten._scaled_dot_product_flash_attention.default(
            *(_flash_operand(value) for value in (q, k, v)), p, causal, False, scale=self.scale)
        # Keep logical output and saved-QKV shapes in the native ABI. Padding
        # belongs only to the private flash invocation; its scale still comes
        # from the original Q/K head dimension or the explicit user override.
        return (values[0][..., :v.shape[-1]], *values[1:])

    def _flash_backward(self, gradient, q, k, v, output, lse, cu_q, cu_k, p, causal, seed, offset):
        values = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
            *(_flash_operand(value) for value in (gradient, q, k, v, output)),
            lse, cu_q, cu_k, q.shape[-2], k.shape[-2], p, causal, seed, offset, scale=self.scale)
        return tuple(value[..., :operand.shape[-1]] for value, operand in zip(values, (q, k, v)))

    def _forward(self, q, k, v, mask):
        q, k, v = (value.to(self.output_dtype) for value in (q, k, v))
        p, causal = self.options["dropout_p"], self.options["is_causal"]
        empty = q.new_empty((0,))
        lse = probability = drop_probability = drop_mask = empty
        seed = offset = cu_q = cu_k = None
        # Numerical forward runs below the region's autograd bridge. Preserve
        # the semantic gradient requirements when selecting native SDPA: in
        # particular a differentiable bias must not select a bias-less VJP.
        probes = tuple(value.detach().requires_grad_(needed) if value is not None else None
                       for value, needed in zip((q, k, v, mask), self.needs))
        with torch.enable_grad():
            implementation = int(torch.ops.aten._fused_sdp_choice.default(*probes, p, causal,
                scale=self.options["scale"], enable_gqa=self.options["enable_gqa"]))
        if self.needs[3] and (self.options["backend"] == "flash_attn_2" or implementation not in
                              (int(SDPBackend.MATH), int(SDPBackend.EFFICIENT_ATTENTION))):
            raise NotImplementedError("Selected native SDPA backend does not provide additive-mask gradients")
        if self.needs[3] and (self.options["backend"] == "flash_attn_2" or implementation not in (
                int(SDPBackend.MATH), int(SDPBackend.EFFICIENT_ATTENTION))):
            raise NotImplementedError("Selected native SDPA backend has no additive-mask gradient contract")
        if self.options["backend"] == "flash_attn_2":
            from deepopt.opt.nn.attention_backend import _validate_and_cast, _kernels

            q, k, v = _validate_and_cast(q, k, v, mask, causal, self.options["enable_gqa"], p, self.scale)
            q, k, v = (value.transpose(1, 2) for value in (q, k, v))
            output, lse, _, seed = _kernels()[0](q, k, v, p, self.scale, False, -1, -1, 0., None, False)
            output = output.transpose(1, 2)
            implementation = _FLASH_ATTN_2
        elif implementation == int(SDPBackend.FLASH_ATTENTION) and q.device.type == "cpu":
            output, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
                q, k, v, p, causal, attn_mask=mask, scale=self.options["scale"])
            implementation = _CPU_FLASH_ATTENTION
        elif implementation == int(SDPBackend.FLASH_ATTENTION):
            output, lse, cu_q, cu_k, _, _, seed, offset, _ = self._flash_forward(q, k, v, p, causal)
        elif implementation == int(SDPBackend.EFFICIENT_ATTENTION):
            output, lse, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                q, k, v, mask, True, p, causal, scale=self.options["scale"])
        elif implementation == int(SDPBackend.OVERRIDEABLE):
            raise NotImplementedError("Overrideable SDPA requires an explicit native AutoDiff VJP")
        elif implementation == int(SDPBackend.CUDNN_ATTENTION):
            output, lse, cu_q, cu_k, _, _, seed, offset, _ = torch.ops.aten._scaled_dot_product_cudnn_attention.default(
                q, k, v, mask, True, p, causal, False, scale=self.options["scale"])
        elif implementation == int(SDPBackend.MATH):
            q, k, v = (value.to(torch.float64 if value.dtype == torch.float64 else torch.float32) for value in (q, k, v))
            if self.options["enable_gqa"]:
                k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
                v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
            q = q * (-sqrt(abs(self.scale)) if self.scale < 0 else sqrt(abs(self.scale)))
            k = k.transpose(-2, -1) * sqrt(abs(self.scale))
            qshape, kshape, vshape = self._math_shapes()
            # Public matmul saves expanded operands. Packing before expansion
            # changes affine groups when a broadcast period is not a multiple
            # of the codec group size, even though forward values are equal.
            q, k = q.expand(qshape), k.expand(kshape)
            scores = q @ k
            if causal:
                allowed = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
                scores = scores.masked_fill(~allowed, float("-inf"))
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
            probability = torch.ops.aten._safe_softmax.default(scores, -1)
            if p:
                drop_probability, drop_mask = torch.ops.aten.native_dropout.default(probability, p, True)
            else:
                drop_probability = probability
            v = v.expand(vshape)
            output = (drop_probability @ v).to(self.output_dtype)
            implementation = int(SDPBackend.MATH)
        else:
            raise NotImplementedError(f"Unsupported native AutoDiff SDPA implementation: {implementation}")
        saved = [*self._pack(q, empty), *self._pack(k, empty), *self._pack(v, empty),
                 lse, probability, drop_probability, drop_mask, seed, offset, cu_q, cu_k,
                 torch.tensor(implementation, dtype=torch.int64, device="cpu")]
        self._validate_saved(implementation, saved)
        return (output, *saved)

    def gradients(self, gradient, output, mask, *saved):
        implementation = int(saved[-1].item())
        self._validate_saved(implementation, saved)
        if implementation == int(SDPBackend.OVERRIDEABLE):
            raise NotImplementedError("Overrideable SDPA requires an explicit native AutoDiff VJP")
        if implementation not in (int(SDPBackend.MATH), int(SDPBackend.FLASH_ATTENTION),
                                  int(SDPBackend.EFFICIENT_ATTENTION), int(SDPBackend.CUDNN_ATTENTION),
                                  _FLASH_ATTN_2, _CPU_FLASH_ATTENTION):
            raise NotImplementedError(f"Unsupported native AutoDiff SDPA implementation: {implementation}")
        qshape, kshape, vshape = (metadata.shape for metadata in self.metas[:3])
        dtype = self.output_dtype
        if implementation == _FLASH_ATTN_2:
            qshape, kshape, vshape = (tuple((shape[0], shape[2], shape[1], shape[3])) for shape in (qshape, kshape, vshape))
        elif implementation == int(SDPBackend.MATH):
            dtype = torch.float64 if dtype == torch.float64 else torch.float32
            qshape, kshape, vshape = self._math_shapes()
        q, k, v = (self._restore(saved[index:index + 3], shape, dtype)
                   for index, shape in zip((0, 3, 6), (qshape, kshape, vshape)))
        lse, probability, drop_probability, drop_mask, seed, offset, cu_q, cu_k = saved[9:-1]
        p, causal = self.options["dropout_p"], self.options["is_causal"]
        bias_gradient = None
        if implementation == _CPU_FLASH_ATTENTION:
            values = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward.default(
                gradient, q, k, v, output, lse, p, causal, attn_mask=mask, scale=self.options["scale"])
        elif implementation == int(SDPBackend.FLASH_ATTENTION):
            values = self._flash_backward(gradient, q, k, v, output, lse, cu_q, cu_k, p, causal, seed, offset)
        elif implementation == int(SDPBackend.EFFICIENT_ATTENTION):
            values = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
                gradient, q, k, v, mask, output, lse, seed, offset, p, list(self.needs), causal,
                scale=self.options["scale"])
            values, bias_gradient = values[:3], values[3]
        elif implementation == int(SDPBackend.CUDNN_ATTENTION):
            values = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(
                gradient, q, k, v, output, lse, seed, offset, mask, cu_q, cu_k, q.shape[-2], k.shape[-2],
                p, causal, scale=self.options["scale"])
        elif implementation == _FLASH_ATTN_2:
            from deepopt.opt.nn.attention_backend import _kernels, _gradient_buffers

            buffers = _gradient_buffers((q, k, v), tuple(value.stride() for value in (q, k, v)))
            _kernels()[1](gradient.transpose(1, 2).contiguous(), q, k, v, output.transpose(1, 2), lse,
                *buffers, p, self.scale, False, -1, -1, 0., None,
                torch.are_deterministic_algorithms_enabled(), rng_state=seed)
            values = tuple(value.transpose(1, 2) for value in buffers)
        else:
            g = gradient.to(dtype)
            dp = g @ v.transpose(-2, -1)
            if p:
                dp = torch.ops.aten.native_dropout_backward.default(dp, drop_mask, 0.0 if p == 1 else 1.0 / (1.0 - p))
            ds = probability * (dp - (dp * probability).sum(-1, keepdim=True))
            bias_gradient = ds.sum_to_size(mask.shape) if mask is not None and self.needs[3] else None
            values = (ds @ k.transpose(-2, -1) * (-sqrt(abs(self.scale)) if self.scale < 0 else sqrt(abs(self.scale))),
                      (q.transpose(-2, -1) @ ds).transpose(-2, -1) * sqrt(abs(self.scale)),
                      drop_probability.transpose(-2, -1) @ g)
            if self.options["enable_gqa"]:
                reduced = [values[0]]
                for value, metadata in zip(values[1:], self.metas[1:3]):
                    shape = metadata.shape
                    # Matmul may expand a singleton batch before the head
                    # repeat is undone. Collapse heads in that expanded batch,
                    # then reduce broadcast axes in the accumulator dtype.
                    reduced.append(value.reshape(value.shape[0], shape[1],
                        value.shape[1] // shape[1], *value.shape[2:]).sum(2))
                values = tuple(reduced)
            values = tuple(value.sum_to_size(metadata.shape)
                           for value, metadata in zip(values, self.metas[:3]))
        values = (*values, bias_gradient)
        return result([value.to(metadata.dtype) for value, metadata, needed in zip(values, self.metas, self.needs) if needed])


def vjp(ctx, node, gradients):
    spec = node.specification
    mask = node.args[3]
    operands = (gradients[0].value, ctx.saved(node.values[0]).value,
                ctx.saved(mask).value if mask is not None else None,
                *(ctx.saved(value).value for value in node.values[1:]))
    from .activation_saving import SavedUse

    ctx.record_residual(node.values[0], node.values, representation="attention_state",
        component_roles=("output", *spec.names),
        recipe={"q_shape": spec.metas[0].shape, "k_shape": spec.metas[1].shape, "v_shape": spec.metas[2].shape,
                "output_dtype": str(spec.output_dtype), "bits": spec.bits,
                "group_size": spec.compression["group_size"] if spec.compression else 1024,
                "backend": spec.options["backend"], "dropout_p": spec.options["dropout_p"],
                "mask_shape": spec.metas[3].shape if spec.metas[3] is not None else None,
                "enable_gqa": spec.options["enable_gqa"]},
        use=SavedUse(role="attention.native_backward"), retention_reason="native_sdpa_backend_contract")
    targets = tuple(value for value, needed in zip(node.args, spec.needs) if needed)
    values = tuple(TensorValue(f"{node.values[0].name}_gradient_{index}", replace(value.meta, requires_grad=False))
                   for index, value in enumerate(targets))
    ctx.builder.graph.add(CallNode("attention.sdpa_vjp", spec.gradients, operands, {}, values))
    return tuple(GradWrapper(target, value) for target, value in zip(targets, values))


def register_rules(registry):
    if not registry.has_backward_rule(SDPA.op):
        registry.register_backward_rule(SDPA.op, vjp, multi_output=True)
