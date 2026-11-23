"""Native VJPs and explicit saved state for prepared spatial operations."""

from dataclasses import replace
from importlib import import_module
from math import prod
import os

import torch

from ..ir import AuxiliaryTensorMeta, CallNode, InputNode, TensorMeta, TensorValue, model_state_origin
from .backward_rules import GradWrapper
from .semantic import result, emit, recorded
from deepopt.opt.backends._shared.normalization import saved_normalization_variant
from deepopt.opt.backends._shared.pooling import saved_pooling_kernel


class Spatial:
    op = "autodiff_spatial"

    def __init__(self, node, context):
        from deepopt.opt._execution import backward_finite_check_policy
        from deepopt.opt.backends._shared.normalization import normalization_variant, normalization_tuning
        from deepopt.opt.backends._shared.pooling import pooling_kernel_variant

        self.module = node.module
        self.bf16_policy = node.bf16_policy
        self.plan = node.plan if node.plan is not None else node.module.plan
        self.context = context
        self.kernel_policy = context.kernel_policy
        self.module_path, self.node_name = node.module_path, node.node_name
        self.precision = "fp8_spatial" if "fp8_training" in self.plan else "native"
        self.family = "convolution" if "conv" in node.op else "normalization" if "normalization" in node.op else "pooling"
        self.profile_ranges = os.environ.get("DEEPOPT_COMPILER_PROFILE_RANGES") == "1"
        self.gradient_check = backward_finite_check_policy()
        self.normalization_variant, self.normalization_tuning = normalization_variant(), normalization_tuning()
        self.pooling_variant = pooling_kernel_variant()
        self.metas = tuple(value.meta if value is not None else None for value in node.args)
        self.training = node.values[0].meta.requires_grad
        self.needs = tuple(self.training and metadata is not None and metadata.requires_grad for metadata in self.metas)
        self.output_dtype = node.values[0].meta.storage_dtype or node.values[0].meta.dtype
        self.logical_dtype = node.values[0].meta.dtype
        self.public_count = len(node.values)
        self.compression = dict(self.plan.get("fp8_training", self.plan)["activation_compression"])
        self.backend = self.plan.get("fp8_training", {}).get("backend", context.kernel_policy.backend)
        if torch.device(self.metas[0].device).type != "cuda":
            self.backend = "reference"

    def lower(self, node):
        values = node.values + tuple(TensorValue(f"{node.node_name}__{name}", AuxiliaryTensorMeta()) for name in self.state_names)
        return CallNode(self.op, recorded(self, self.forward, "forward"), node.args, node.kwargs, values, self)

    def inference(self, *inputs, **kwargs):
        output = self.forward(*inputs, **kwargs)
        return result(list(output[:self.public_count]))


class Convolution(Spatial):
    state_names = ("saved_source", "codec_step", "codec_minimum")

    def __init__(self, node, context):
        super().__init__(node, context)
        from deepopt.opt.nn.convolution.geometry import ConvGeometry
        from deepopt.opt.nn.convolution_transpose.geometry import ConvTransposeGeometry
        from deepopt.opt.backends._shared.activation_codec import convolution_codec
        from deepopt.opt.nn.convolution.geometry import _spatial_tuple

        module = self.module
        self.transposed = node.op == "deepopt_conv_transpose"
        self.rank = getattr(module, "rank", len(self.metas[1].shape) - 2)
        options = node.options
        def spatial_option(name, default, minimum):
            return _spatial_tuple(options.get(name, getattr(module, name, default)), self.rank, name, minimum=minimum)
        self.unbatched = len(self.metas[0].shape) == self.rank + 1
        shape = (1, *self.metas[0].shape) if self.unbatched else self.metas[0].shape
        self.original_shape = shape
        self.pad_mode, self.explicit_padding = options.get("padding_mode", "zeros"), options.get("explicit_padding")
        self.synthetic_right = (0,) * self.rank
        raw_padding = options.get("padding", getattr(module, "padding", 0))
        stride, dilation = spatial_option("stride", 1, 1), spatial_option("dilation", 1, 1)
        if self.explicit_padding:
            spatial = tuple(size + self.explicit_padding[2 * (self.rank - axis - 1)] + self.explicit_padding[2 * (self.rank - axis - 1) + 1]
                            for axis, size in enumerate(shape[2:]))
            shape, padding = (*shape[:2], *spatial), (0,) * self.rank
        elif isinstance(raw_padding, str):
            if raw_padding == "valid":
                padding = (0,) * self.rank
            elif raw_padding == "same" and all(step == 1 for step in stride):
                total = tuple(dil * (kernel - 1) for dil, kernel in zip(dilation, self.metas[1].shape[2:]))
                self.synthetic_right = tuple(value % 2 for value in total)
                shape = (*shape[:2], *(size + extra for size, extra in zip(shape[2:], self.synthetic_right)))
                padding = tuple(value // 2 for value in total)
            else:
                raise ValueError("Invalid or strided SAME convolution padding")
        else:
            padding = spatial_option("padding", 0, 0)
        self.geometry = (ConvTransposeGeometry if self.transposed else ConvGeometry)(shape, self.metas[1].shape,
            stride, padding, dilation,
            options.get("groups", getattr(module, "groups", 1)),
            **({"output_padding": spatial_option("output_padding", 0, 0)} if self.transposed else {"synthetic_padding_right": self.synthetic_right}))
        self.codec = convolution_codec()
        device = torch.device(self.metas[0].device).type
        self.compute_dtype = (torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device)
                              and self.metas[0].dtype != torch.float64 else self.metas[0].dtype)
        self.compressed = self.needs[1] and self.compression["enabled"] and prod(shape) > 0
        self.input_format = self.memory_format(self.metas[0])
        self.weight_format = self.memory_format(self.metas[1])
        self.kernels = import_module("deepopt.opt.backends.native." + ("convolution_transpose" if self.transposed else "convolution"))

    def lower(self, node):
        """Keep the native forward separate from explicit source-pack IR."""
        if type(self) is not Convolution:
            # FP8 convolution owns a different seven-slot state ABI.
            return super().lower(node)
        return CallNode(self.op, recorded(self, self.forward, "forward"), node.args, node.kwargs,
                        node.values, self)

    def fresh_dx_contract(self):
        """Prove the narrow uncropped NCHW dX allocation used by donation.

        dx_db explicitly makes this native result contiguous after its dtype
        projection. No input/view storage can escape through this geometry.
        Other convolution routes retain unknown ownership/layout metadata.
        """
        if (not getattr(self.context, "donate_fresh_backward_add", False)
                or type(self) is not Convolution or not self.needs[0]
                or self.rank != 2 or self.transposed or self.unbatched
                or self.explicit_padding or any(self.synthetic_right)
                or self.pad_mode != "zeros" or self.input_format != torch.contiguous_format
                or self.geometry.groups != 1 or self.geometry.kernel_size != (1, 1)
                or self.geometry.stride != (1, 1) or self.geometry.dilation != (1, 1)
                or self.geometry.padding != (0, 0)
                or any(size <= 0 for size in self.metas[0].shape)):
            return None
        from ..ir import FreshAllocationContract
        from ..trace.metadata_utils import contiguous_stride

        metadata = replace(self.metas[0], requires_grad=False,
                           stride=contiguous_stride(self.metas[0].shape))
        return FreshAllocationContract(metadata, "convolution.native_dx_db")

    def inference(self, *inputs, **kwargs):
        if type(self) is Convolution:
            return self.forward(*inputs, **kwargs)
        return super().inference(*inputs, **kwargs)

    @staticmethod
    def memory_format(metadata):
        if metadata.stride is not None and len(metadata.shape) in (4, 5):
            from torch._prims_common import suggest_memory_format

            # A singleton 1x1 kernel can have channel stride one while native
            # eager execution still selects contiguous layout. Choosing a
            # different gradient layout changes native reduction rounding.
            descriptor = torch.empty_strided(metadata.shape, metadata.stride,
                                              dtype=metadata.dtype, device="meta")
            return suggest_memory_format(descriptor)
        return torch.contiguous_format

    def source(self, x):
        x = x.unsqueeze(0) if self.unbatched else x
        if self.explicit_padding:
            return torch.nn.functional.pad(x, self.explicit_padding, mode=self.pad_mode)
        if any(self.synthetic_right):
            return torch.nn.functional.pad(x, tuple(value for extra in reversed(self.synthetic_right) for value in (0, extra)))
        return x

    def restore_input_gradient(self, gradient):
        if any(self.synthetic_right):
            gradient = gradient[(slice(None), slice(None), *(slice(0, size) for size in self.original_shape[2:]))]
        if self.explicit_padding:
            from deepopt.opt.nn.padding import PaddingGeometry, padding_adjoint

            gradient = padding_adjoint(gradient, PaddingGeometry(self.original_shape,
                tuple(self.explicit_padding), self.pad_mode, gradient.dtype, self.input_format))
        return gradient.squeeze(0) if self.unbatched else gradient

    def forward(self, x, weight, bias):
        from deepopt.opt.quantization.activation_tensor import FP8Activation

        if isinstance(x, FP8Activation):
            x = x.dequantize(x.dtype)

        source = self.source(x)
        amp = torch.autocast(x.device.type, enabled=torch.is_autocast_enabled(x.device.type),
                             dtype=torch.get_autocast_dtype(x.device.type), cache_enabled=False)
        # The joint training Function runs its forward with grad mode off.
        # Use the captured gradient contract for kernel selection so validation
        # stays in no-grad and cannot reenter eager convolution autograd.
        with amp, torch.set_grad_enabled(self.training), torch._C._AutoDispatchBelowAutograd():
            output = self.kernels.native_convolution(source, weight, bias, self.geometry)
        self.module.last_execution = {"forward": "native", "backward": "native_autodiff",
            "compression": self.compressed, "saved_state_bytes": 0}
        output = output.to(self.logical_dtype)
        return output.squeeze(0) if self.unbatched else output

    def pack_source(self, x):
        from deepopt.opt.nn.convolution import source_matrix
        from deepopt.opt.quantization.activation import quantize_pack_activation, _quantize_pack_strided_activation
        from deepopt.opt.quantization.activation_tensor import FP8Activation

        x = x.dequantize(x.dtype) if isinstance(x, FP8Activation) else x
        source = self.source(x)
        if self.compressed:
            packed = (quantize_pack_activation(source_matrix(source).to(self.compute_dtype),
                self.compression["group_size"], bits=self.compression["bits"])
                if self.transposed or self.codec == "materialized" else _quantize_pack_strided_activation(
                    source.movedim(1, -1), self.compression["group_size"], bits=self.compression["bits"],
                    source_dtype=self.compute_dtype))
            saved = packed.data, packed.step, packed.minimum
        else:
            saved = (source,)
        self.module.last_execution["saved_state_bytes"] = sum(value.numel() * value.element_size() for value in saved)
        return result(saved)

    def saved_source(self, ctx, node):
        """Insert a typed, invocation-local source pack requested only by dW."""
        from types import MappingProxyType
        from .activation_saving import SavedUse

        metadata = self.metas[0]
        if self.compressed:
            from deepopt.opt.quantization.activation import _dimensions

            groups, _, words = _dimensions(self.geometry.input_matrix_shape,
                self.compression["group_size"], None, self.compression["bits"])
            descriptions = (((groups, words), torch.int32, "packed"),
                            ((groups,), torch.float32, "dense"), ((groups,), torch.float32, "dense"))
        else:
            descriptions = ((self.geometry.input_shape, metadata.dtype, "dense"),)
        state = tuple(TensorValue(f"{node.values[0].name}__{name}", TensorMeta(shape, dtype,
            metadata.device, requires_grad=False, representation=representation))
            for name, (shape, dtype, representation) in zip(self.state_names, descriptions))
        recipe = {"bits": self.compression["bits"], "group_size": self.compression["group_size"],
            "logical_shape": metadata.shape, "logical_dtype": metadata.dtype, "logical_stride": metadata.stride,
            "restored_stride": metadata.stride, "representation_role": "convolution_source",
            "source_shape": self.geometry.input_shape, "matrix_shape": self.geometry.input_matrix_shape,
            "traversal": "channels", "codec": self.codec, "rounding_domain": "forward_compute_dtype",
            "source_cast": self.compute_dtype, "unbatched": self.unbatched, "transposed": self.transposed,
            "bf16_policy": self.bf16_policy.as_dict(),
            "explicit_padding": tuple(self.explicit_padding or ()), "padding_mode": self.pad_mode,
            "synthetic_padding_right": self.synthetic_right}
        packed = CallNode("codec.save_convolution_source" if self.compressed else "convolution.save_source",
            recorded(self, self.pack_source, "forward"), (node.args[0],), {}, state, MappingProxyType(recipe))
        packed.bf16_policy = self.bf16_policy
        graph = ctx.forward_graph
        index = graph.nodes.index(node) + 1
        graph.add(packed)
        graph.nodes.pop()
        graph.nodes.insert(index, packed)
        saved = tuple(ctx.saved(value) for value in state)
        ctx.record_residual(node.args[0], state,
            representation="packed_convolution_source" if self.compressed else "exact_convolution_source",
            component_roles=("payload", "codec_step", "codec_minimum") if self.compressed else ("value",),
            use=SavedUse(consumer=node.op, role="weight_gradient_coefficient",
                         allow_approximation=self.compressed, rounding_domain="forward_compute_dtype"),
            recipe=recipe, retention_reason="Requested convolution weight gradient")
        return saved if self.compressed else (*saved, None, None)

    def dx_db(self, gradient, weight, *, cuda_geometry=False):
        from deepopt.opt.backends.native.convolution import native_input_bias_gradients

        gradient = (gradient.unsqueeze(0) if self.unbatched else gradient).to(self.compute_dtype)
        dx, _, db = native_input_bias_gradients(gradient, weight.to(self.compute_dtype) if self.needs[0] else None,
            self.geometry, (self.needs[0], False, self.needs[2]), input_memory_format=self.input_format,
            weight_memory_format=self.weight_format, transposed=self.transposed, execution_info=self.module.last_execution,
            **({"cuda_geometry": True} if cuda_geometry else {}))
        if self.needs[0]:
            dx = dx.to(self.metas[0].dtype).contiguous(memory_format=self.input_format)
            dx = self.restore_input_gradient(dx)
        return result([value for value, needed in ((dx, self.needs[0]),
            (db.to(self.metas[2].dtype) if db is not None else None, self.needs[2])) if needed])

    def dw(self, gradient, data, step, minimum):
        from deepopt.opt.nn.convolution import restore_matrix
        from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation, _dequantize_unpack_spatial_activation

        gradient = (gradient.unsqueeze(0) if self.unbatched else gradient).to(self.compute_dtype)
        source = data
        if self.compressed:
            packed = PackedActivation(data, step, minimum, self.geometry.input_matrix_shape,
                self.compression["group_size"], bits=self.compression["bits"])
            source = (restore_matrix(dequantize_unpack_activation(packed, dtype=self.compute_dtype),
                self.geometry.input_shape, self.input_format) if self.transposed or self.codec == "materialized" else
                _dequantize_unpack_spatial_activation(packed, self.geometry.input_shape,
                    dtype=self.compute_dtype, memory_format=self.input_format))
            if not self.transposed:
                from deepopt.opt.nn.convolution.geometry import zero_synthetic_padding

                zero_synthetic_padding(source, self.geometry)
        return self.kernels.native_weight_gradient(source.to(self.compute_dtype), gradient, self.geometry,
            weight_memory_format=self.weight_format).to(self.metas[1].dtype).contiguous(memory_format=self.weight_format)

    def reuse_gradients(self, gradient, weight, data, step, minimum):
        """Full dW followed by bounded dX into a fresh private decode result."""
        from contextlib import nullcontext
        from deepopt.opt.backends.native.convolution.ops import _native_full_dw_tiled_dx
        from deepopt.opt.quantization.activation import PackedActivation, _dequantize_unpack_spatial_activation

        if (self.context.native_conv_workspace_reuse != "tiled_dx" or not self.compressed or self.codec == "materialized"
                or self.compute_dtype != torch.bfloat16 or self.metas[0].dtype != torch.bfloat16
                or not self.needs[0] or not self.needs[1] or self.transposed or self.unbatched
                or self.explicit_padding or any(self.synthetic_right) or self.pad_mode != "zeros"
                or torch.is_grad_enabled()):
            raise ValueError("Private convolution workspace runtime contract changed")
        packed = PackedActivation(data, step, minimum, self.geometry.input_matrix_shape,
            self.compression["group_size"], bits=self.compression["bits"])
        profiling = self.profile_ranges or torch.autograd._profiler_enabled()
        with torch.profiler.record_function("deepopt.convolution.private_workspace.decode") if profiling else nullcontext():
            source = _dequantize_unpack_spatial_activation(packed, self.geometry.input_shape,
                dtype=self.compute_dtype, memory_format=self.input_format)
        if any(source.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
               for value in (gradient, weight, data, step, minimum)):
            raise ValueError("Private convolution decoder returned aliased storage")
        dx, dw, db = _native_full_dw_tiled_dx(source, gradient, weight.to(self.compute_dtype), self.geometry,
            batch_size=self.context.native_conv_dx_batch_size, need_bias=self.needs[2],
            execution_info=self.module.last_execution, profile_ranges=self.profile_ranges)
        # Match the old public gradient tuple: dx[,db] then weight gradient.
        return result([dx, *([db.to(self.metas[2].dtype)] if self.needs[2] else []),
            dw.to(self.metas[1].dtype).contiguous(memory_format=self.weight_format)])


class Normalization(Spatial):
    state_names = ("saved_xhat", "codec_step", "codec_minimum", "inverse", "mean", "weight", "center_shift", "gradient_selector")

    def __init__(self, node, context):
        super().__init__(node, context)
        from deepopt.opt.nn.normalization.geometry import NormGeometry
        from deepopt.opt.nn.normalization.modules import DOBatchNorm2d
        from deepopt.opt.nn.normalization.policy import validate_normalization_backend

        validate_normalization_backend(self.bf16_policy, self.backend, torch.device(self.metas[0].device).type)

        self.options = node.options
        self.kind = self.options.get("kind", "batch" if type(self.module) is DOBatchNorm2d else "group")
        self.empty = prod(self.metas[0].shape) == 0
        self.use_batch = self.options.get("training", self.module.training or node.args[3] is None) if self.kind == "batch" else True
        self.geometry = NormGeometry(self.kind, self.metas[0].shape, self.options.get("groups", getattr(self.module, "num_groups", 1)),
                                     self.options.get("eps", getattr(self.module, "eps", 1e-5)), self.use_batch)
        self.need_h = not self.empty and (self.needs[1] or self.needs[0] and self.geometry.batch_stats)
        self.compression["enabled"] &= self.need_h
        if self.compression["enabled"]:
            self.compression["group_size"] = min(self.compression["group_size"], self.geometry.elements)
        from .gradient import GradientPolicy

        self.policy = GradientPolicy(self.module, node.root or self.module,
            not isinstance(node.args[0].producer, InputNode), context.callsite_occurrence)
        self.output_layout = node.values[0].meta.layout if node.values[0].meta.storage_dtype is not None else None
        self.scaling_state = getattr(self.module, "fp8_scaling", None) if self.output_layout is not None else None
        self.state_training = self.module.training and torch.is_grad_enabled()
        self.empty_selector = torch.tensor(-1, device="cpu", dtype=torch.int64)
        # Eager compression-off GroupNorm dispatches to native ATen, including
        # prepared DOGroupNorm. Its saved statistics can be BF16 rather than
        # the FP32 statistics of the compressed normalized-value recipe.
        self.native_group = None
        if (not self.bf16_policy.storage_enabled and not self.empty and self.kind == "group" and "fp8_training" not in self.plan
                and not self.plan["activation_compression"]["enabled"]):
            from copy import copy
            from .library_group_norm import LibraryGroupNorm

            native_node = node
            if not hasattr(self.module, "num_groups"):
                # Functional adapters carry groups/eps on the call, not the
                # shared module. Keep those immutable per-call options local.
                native_node = copy(node)
                native_node.module = copy(self.module)
                native_node.module.num_groups = self.geometry.groups
                native_node.module.eps = self.geometry.eps
            self.native_group = LibraryGroupNorm(native_node, context)

    def lower(self, node):
        # Native dense normalization has a known state ABI. Keep backend-owned
        # FP8 transport descriptors opaque, but expose every BF16 payload and
        # exact auxiliary shape for saved-state coverage and byte accounting.
        if self.output_layout is not None or self.policy.enabled:
            return super().lower(node)
        from .semantic import meta, packed_metas

        source = self.metas[0]
        empty = meta((0,), source.dtype, source.device)
        statistics = meta((self.geometry.domains,), torch.float32, source.device)
        packed = self.training and self.compression["enabled"]
        if packed:
            data, step, minimum = packed_metas(self.geometry.matrix_shape, self.compression,
                source.device, (1, self.compression["group_size"]))
        else:
            data = replace(source, requires_grad=False) if self.training and self.need_h else empty
            step = minimum = empty
        inverse = statistics if self.training and (self.need_h or self.needs[0]) else empty
        mean = statistics if self.training and self.need_h and not packed else empty
        weight = (replace(self.metas[1], requires_grad=False)
                  if self.training and self.needs[0] and self.metas[1] is not None else empty)
        center = statistics if packed and self.compression.get("norm_xhat_recenter", False) else empty
        selector = meta((), torch.int64, "cpu")
        descriptions = (data, step, minimum, inverse, mean, weight, center, selector)
        if self.empty:
            descriptions = (*([empty] * 7), selector)
        if self.native_group is not None:
            native = self.native_group.lower(node)
            saved = dict(zip(self.native_group.state_names, native.values[self.public_count:]))
            data = saved["source"].meta if "source" in saved else empty
            mean = replace(saved["mean"].meta, shape=(self.geometry.domains,), stride=(1,)) if "mean" in saved else empty
            inverse = replace(saved["rstd"].meta, shape=(self.geometry.domains,), stride=(1,)) if "rstd" in saved else empty
            weight = saved["weight"].meta if "weight" in saved else empty
            descriptions = (data, empty, empty, inverse, mean, weight, empty, selector)
        state = tuple(TensorValue(f"{node.node_name}__{name}", metadata)
                      for name, metadata in zip(self.state_names, descriptions))
        if prod(weight.shape) and node.args[1] is not None:
            from ..ir import inherit_model_alias

            if self.native_group is None or not self.native_group.cast_affine or weight.dtype == self.metas[1].dtype:
                inherit_model_alias(state[5], node.args[1])
            else:
                # Native AMP may produce a separate affine cast allocation.
                state[5].provenance = node.args[1].provenance
                state[5].model_state_copy = True
                state[5].model_state_origin = model_state_origin(node.args[1])
                state[5].model_state_transforms = (*node.args[1].model_state_transforms, "to")
        return CallNode(self.op, recorded(self, self.forward, "forward"), node.args, node.kwargs,
                        node.values + state, self)

    def forward(self, x, weight, bias, running_mean, running_var, counter, *, output=None):
        from deepopt.opt.nn.autograd.normalization.function import NormalizationFunction
        from .stateful_fp8 import ForwardResiduals

        if output is not None and (self.native_group is not None or self.empty or self.output_layout is not None
                                   or self.policy.enabled or self.kind != "group"):
            raise ValueError("Compiler-private normalization destination requires the qualified dense group route")
        if self.native_group is not None:
            return self.native_group_forward(x, weight, bias, running_mean, running_var, counter)
        factor = self.options.get("momentum", getattr(self.module, "momentum", 0.1))
        if self.kind == "batch" and counter is not None and self.module.training and self.module.track_running_stats:
            counter.add_(1)
            if factor is None:
                factor = counter.to(torch.float32).reciprocal()
        factor = 0.0 if factor is None else factor
        if self.empty:
            # Eager accepts empty BN and N=0 GN without evaluating moments or
            # updating running statistics. Only BN's invocation counter above
            # changes. No activation/statistic/affine payload is a derivative
            # dependency: every requested gradient is an empty/zero tensor.
            output = torch.empty_like(x, dtype=self.output_dtype)
            self.module.last_execution = dict(backend="native_empty", compression=False,
                                             saved_state_bytes=0)
            empty = x.new_empty((0,))
            return output, *([empty] * 7), self.empty_selector
        selector = self.policy.select(x)
        empty = x.new_empty((0,))
        ctx = ForwardResiduals(self.needs)
        self.module.last_execution = info = {}
        output = NormalizationFunction.forward(ctx, x, weight, bias, running_mean, running_var, self.geometry,
            self.plan, self.output_layout, self.scaling_state, self.state_training, factor, self.training, info,
            bf16_policy=self.bf16_policy, output=output)
        saved = ctx.tensors if self.training else (None,) * 7
        return output, *(value if value is not None else empty for value in saved), selector if selector is not None else self.empty_selector

    def native_group_forward(self, *inputs):
        """Keep the spatial residual slots while using eager's native recipe."""
        native = self.native_group
        values = native.forward(*inputs)
        values = values if isinstance(values, tuple) else (values,)
        state = dict(zip(native.state_names, values[1:]))
        empty = inputs[0].new_empty((0,))
        selector = self.policy.select(inputs[0])
        saved = (state.get("source", empty), empty, empty,
                 state["rstd"].reshape(-1) if "rstd" in state else empty,
                 state["mean"].reshape(-1) if "mean" in state else empty,
                 state.get("weight", empty), empty,
                 selector if selector is not None else self.empty_selector)
        self.module.last_execution = dict(backend="native", compression=False,
            saved_state_bytes=sum(value.numel() * value.element_size() for value in saved))
        return values[0], *saved

    def native_group_gradients(self, gradient, *state):
        native = self.native_group
        data, _, _, inverse, mean, weight, _, selector = state
        statistics_shape = (native.n, native.groups)
        binding = dict(source=data, mean=mean.reshape(statistics_shape) if "mean" in native.state_names else mean,
                       rstd=inverse.reshape(statistics_shape) if "rstd" in native.state_names else inverse, weight=weight)
        values = native.gradients(gradient, *(binding[name] for name in native.state_names))
        values = list(values if isinstance(values, tuple) else (values,))
        slot = self.policy.slot(selector)
        if self.needs[0] and slot is not None:
            with slot.output(self.geometry.shape, self.metas[0].dtype, training=self.state_training,
                             check_finite=self.gradient_check) as output:
                values[0] = output.convert(values[0])
        return result(values)

    @saved_normalization_variant
    def gradients(self, gradient, *state):
        from deepopt.opt.backends.normalization import executor
        from deepopt.opt.quantization.activation import PackedActivation
        from deepopt.opt.quantization.activation_tensor import FP8Activation

        if self.empty:
            return result([torch.zeros(metadata.shape, dtype=metadata.dtype, device=gradient.device)
                           for metadata, needed in zip(self.metas[:3], self.needs[:3]) if needed])
        if self.native_group is not None:
            return self.native_group_gradients(gradient, *state)
        data, step, minimum, inverse, mean, weight, center, selector = state
        packed = original = None
        if self.compression["enabled"]:
            group = self.compression["group_size"]
            packed = PackedActivation(data, step, minimum, self.geometry.matrix_shape, group, (1, group), self.compression["bits"])
        elif data.numel():
            original = (FP8Activation(data, step, self.metas[0].shape, self.metas[0].block_shape,
                self.metas[0].layout, self.metas[0].dtype, self.metas[0].stride)
                if self.metas[0].storage_dtype is not None else data)
        arguments = dict(packed=packed,
            x=original, mean=mean if mean.numel() else None,
            inverse=inverse if inverse.numel() else None, center_shift=center if center.numel() else None,
            input_dtype=self.metas[0].dtype, need_dx=self.needs[0], need_dw=self.needs[1], need_db=self.needs[2],
            execution_info=self.module.last_execution)
        slot = self.policy.slot(selector)
        compute = executor(gradient, self.backend)
        if self.needs[0] and slot is not None:
            arguments["input_dtype"] = torch.float32
            with slot.output(self.geometry.shape, self.metas[0].dtype, training=self.state_training,
                             check_finite=self.gradient_check) as output:
                values = compute.backward(gradient, weight if weight.numel() else None, self.geometry, gradient_output=output, **arguments)
        else:
            values = compute.backward(gradient, weight if weight.numel() else None, self.geometry, **arguments)
        return result([value if isinstance(value, FP8Activation) else value.to(metadata.dtype)
                       for value, metadata, needed in zip(values, self.metas[:3], self.needs[:3]) if needed])


class Pooling(Spatial):
    state_names = ("routing", "coefficient", "codec_step", "codec_minimum", "factor", "samples", "public_guard", "index_guard", "gradient_selector")

    def __init__(self, node, context):
        super().__init__(node, context)
        from deepopt.opt.nn.pooling.geometry import geometry
        from deepopt.opt.nn.activation.geometry import ReLUGeometry
        from torch._subclasses.fake_tensor import FakeTensorMode
        from ..ir import TensorValue
        from .joint import map_tree
        from .gradient import GradientPolicy

        self.options = node.options
        self.layout = node.values[0].meta.layout or ("channels" if Convolution.memory_format(self.metas[0]) != torch.contiguous_format else "last")
        with FakeTensorMode():
            source = torch.empty(self.metas[0].shape, dtype=self.metas[0].dtype, device=self.metas[0].device)
            mapping = {value: torch.empty(value.meta.shape, dtype=value.meta.dtype, device=value.meta.device)
                       for value in node.inputs() if isinstance(value, TensorValue)}
            self.geometry = geometry(self.module.kind, source, map_tree(self.options, mapping), self.layout)
        self.coefficient_layout = self.metas[0].layout if self.metas[0].storage_dtype is not None else "last"
        self.transport = ReLUGeometry(self.metas[0].shape, self.coefficient_layout)
        self.blocks = self.plan.get("block_shapes", {}).get("activation")
        self.coefficient_block = self.metas[0].block_shape if self.metas[0].storage_dtype is not None else self.blocks
        self.scaling_state = getattr(self.module, "fp8_scaling", None) if node.values[0].meta.storage_dtype is not None else None
        self.policy = GradientPolicy(self.module, node.root or self.module,
            not isinstance(node.args[0].producer, InputNode), context.callsite_occurrence)
        self.state_training = self.module.training and torch.is_grad_enabled()
        self.empty_selector = torch.tensor(-1, device="cpu", dtype=torch.int64)
        self.need_state = self.needs[0] and prod(self.metas[0].shape) > 0
        self.global_average = (self.geometry.family == "adaptive_average" and self.geometry.output_elements == 1
                               and self.metas[0].storage_dtype is None and not self.policy.enabled)
        # Eager compression-off pooling delegates to the native operation.
        # Its BF16 overlapping-window accumulation differs from a promoted
        # scatter followed by one final cast. Preserve that native adjoint.
        self.native_dense_max = (self.module.kind == "max_pool2d" and not self.compression["enabled"]
                                 and self.metas[0].storage_dtype is None and self.output_dtype == self.metas[0].dtype)

    def lower(self, node):
        if self.global_average and self.output_dtype == self.metas[0].dtype:
            def forward(x, **options):
                return self.forward(x, **options)[0]
            return CallNode(self.op, recorded(self, forward, "forward"), node.args, node.kwargs,
                            node.values, self)
        return super().lower(node)

    def forward(self, x, **options):
        from deepopt.opt.backends.pooling import executor
        from deepopt.opt.quantization.activation import PackedActivation
        from deepopt.opt.quantization.indices import PackedIndices
        from deepopt.opt.quantization.activation_tensor import FP8Activation
        from deepopt.opt._execution import finite_checks_enabled

        empty = x.new_empty((0,))
        selector = self.policy.select(x)
        if self.global_average and self.output_dtype == x.dtype:
            output = self.geometry.spec.native(x, **options)
            self.module.last_execution = {"forward_variant": "native_global_average", "saved_state_bytes": 0}
            return (output, *(empty for _ in self.state_names))
        samples = options.get("_random_samples")
        if samples is None and self.geometry.family == "fractional_max":
            samples = torch.rand(self.geometry.batch, self.geometry.channels, self.geometry.rank,
                                 device=x.device, dtype=x.dtype)
        arguments = dict(options=options, dtype=self.output_dtype, block_shape=self.blocks,
            compression=self.compression, check_finite=finite_checks_enabled() and (self.output_dtype in
                (torch.float8_e4m3fn, torch.float8_e5m2) or self.compression["enabled"] and self.geometry.family == "power"), need_state=self.need_state,
            input_block_shape=self.blocks, samples=samples)
        compute = executor(x, self.backend)
        if self.scaling_state is not None:
            output, inverse, state, _, _, info = self.scaling_state.activation_forward(
                x, self.geometry, compute, arguments, training=self.state_training)
        else:
            output, inverse, state, _, _, info = compute.forward(x, self.geometry, **arguments)
        routing, coefficient = state.routing, state.coefficient
        saved = (routing.data if isinstance(routing, PackedIndices) else routing,
                 coefficient.data if isinstance(coefficient, PackedActivation) else coefficient,
                 coefficient.step if isinstance(coefficient, PackedActivation) else None,
                 coefficient.minimum if isinstance(coefficient, PackedActivation) else None,
                 state.factor, state.samples, state.public_indices, options.get("indices"),
                 selector if selector is not None else self.empty_selector)
        self.module.last_execution = {**info, "saved_state_bytes": sum(value.numel() * value.element_size()
                                                                      for value in saved if value is not None)}
        public = [FP8Activation(output, inverse, self.geometry.output_shape, self.blocks, self.layout, self.logical_dtype)
                  if inverse is not None else self.geometry.transport.restore(output)]
        if self.public_count > 1:
            public.append(state.public_indices)
        return (*public, *(value if value is not None else empty for value in saved))

    @saved_pooling_kernel
    def dx(self, gradient, *state):
        from deepopt.opt.backends.pooling import executor
        from deepopt.opt.quantization.activation import PackedActivation
        from deepopt.opt.quantization.indices import PackedIndices
        from math import isinf

        if self.global_average and self.output_dtype == self.metas[0].dtype:
            return (gradient.to(self.metas[0].dtype) / self.geometry.input_elements).expand(self.metas[0].shape)
        *state, selector = state
        routing, coefficient, step, minimum, factor, samples, _, _ = (value if value.numel() else None for value in state)
        g = self.geometry
        if self.native_dense_max:
            from torch.nn.modules.utils import _pair
            from deepopt.opt.backends.reference.pooling.windows import decode_locations

            if not prod(g.shape):
                return torch.empty(g.shape, device=gradient.device, dtype=self.metas[0].dtype)
            indices, _ = decode_locations(routing, g, samples)
            # The library backward reads indices and shape/dtype, not source
            # values. A scalar expanded view provides that metadata without
            # retaining or allocating an activation-sized dummy source.
            source = torch.empty((), device=gradient.device, dtype=self.metas[0].dtype).expand(g.shape)
            options = self.options
            kernel = _pair(options["kernel_size"])
            output = torch.ops.aten.max_pool2d_with_indices_backward.default(
                gradient, source, kernel, _pair(options.get("stride") or kernel), _pair(options.get("padding", 0)),
                _pair(options.get("dilation", 1)), options.get("ceil_mode", False), indices.reshape(g.output_shape))
            self.module.last_execution.update(backward_variant="native_dense_max_pool2d", backward_backend="native_torch")
            return output.contiguous(memory_format=Convolution.memory_format(self.metas[0]))
        if routing is not None and self.compression["enabled"]:
            shape = self.transport.matrix_shape if g.family == "unpool" else g.transport.matrix_shape
            routing = PackedIndices(routing, shape, g.route_values)
        if coefficient is not None and self.compression["enabled"] and g.family == "power" and not isinf(g.power):
            coefficient = PackedActivation(coefficient, step, minimum, self.transport.matrix_shape,
                self.compression["group_size"], self.coefficient_block, self.compression["bits"])
        arguments = dict(routing=routing, coefficient=coefficient, factor=factor, samples=samples,
            coefficient_layout=self.coefficient_layout, input_dtype=self.metas[0].dtype, execution_info=self.module.last_execution)
        slot = self.policy.slot(selector)
        if slot is not None:
            with slot.output(g.shape, self.metas[0].dtype, training=self.state_training,
                             check_finite=self.gradient_check) as output:
                return executor(gradient, self.backend).backward(gradient, g, gradient_output=output, **arguments)
        return executor(gradient, self.backend).backward(gradient, g, **arguments)


def specification(node, context):
    if node.op in ("deepopt_convolution", "deepopt_conv_transpose"):
        plan = node.plan if node.plan is not None else node.module.plan
        if "fp8_training" in plan:
            from .convolution import FP8Convolution

            if (not node.op.endswith("conv_transpose") and plan["fp8_training"].get("convolution_policy") == "native_spatial"):
                native = Convolution(node, context)
                if not native.geometry.pointwise or any(block is not None for block in plan["block_shapes"].values()):
                    return native
            return FP8Convolution(node, context)
        return Convolution(node, context)
    if node.op == "deepopt_normalization":
        if type(node.module) is torch.nn.BatchNorm2d:
            from .library_batch_norm import LibraryBatchNorm

            return LibraryBatchNorm(node, context)
        plan = node.plan if node.plan is not None else node.module.plan
        if (not node.bf16_policy.storage_enabled and type(node.module) is torch.nn.GroupNorm
                and "fp8_training" not in plan and not plan["activation_compression"]["enabled"]):
            from .library_group_norm import LibraryGroupNorm

            return LibraryGroupNorm(node, context)
        return Normalization(node, context)
    return Pooling(node, context)


def vjp(ctx, node, gradients):
    spec = node.specification
    saved = (spec.saved_source(ctx, node) if spec.needs[1] else ()) if type(spec) is Convolution else tuple(
        ctx.saved(value) for value in node.values[spec.public_count:])
    if isinstance(spec, Pooling):
        if spec.global_average and spec.output_dtype == spec.metas[0].dtype:
            from .activation_saving import SavedUse

            ctx.record_residual(node.args[0], (), representation="metadata_only",
                use=SavedUse(consumer=node.op, role="input_gradient"),
                recipe={"shape": spec.metas[0].shape, "input_elements": spec.geometry.input_elements},
                retention_reason="Global average adjoint requires geometry only")
        metadata = replace(spec.metas[0], representation="opaque") if spec.policy.enabled else spec.metas[0]
        return (GradWrapper(node.args[0], emit(ctx, "pooling.vjp", spec.dx, (gradients[0], *saved), metadata)),)
    targets = tuple(value for value, needed in zip(node.args[:3], spec.needs[:3]) if needed)
    if not targets:
        return ()
    if isinstance(spec, Normalization):
        from .semantic import record_normalization_state

        record_normalization_state(ctx, node, spec, node.values[spec.public_count:], spec.state_names)
        operands = (gradients[0].value, *(value.value for value in saved))
        values = tuple(TensorValue(f"{node.values[0].name}_grad_{index}", replace(target.meta, requires_grad=False))
                       for index, target in enumerate(targets))
        if spec.needs[0] and spec.policy.enabled:
            values[0].meta = replace(values[0].meta, representation="opaque")
        ctx.builder.graph.add(CallNode("normalization.vjp", recorded(spec, spec.gradients, "backward"), operands, {}, values))
        return tuple(GradWrapper(target, value) for target, value in zip(targets, values))
    contributions = []
    if spec.needs[0] or spec.needs[2]:
        weight = ctx.saved(node.args[1]) if spec.needs[0] else None
        target_values = tuple(node.args[index] for index in (0, 2) if spec.needs[index])
        outputs = tuple(TensorValue(f"{node.values[0].name}_grad_{index}", replace(target.meta, requires_grad=False))
                        for index, target in enumerate(target_values))
        if spec.needs[0]:
            allocation = spec.fresh_dx_contract()
            if allocation is not None:
                outputs[0].meta = allocation.meta
                outputs[0].fresh_allocation = allocation
        ctx.builder.graph.add(CallNode("convolution.native_dx_db", recorded(spec, spec.dx_db, "backward"),
            (gradients[0].value, weight.value if weight is not None else None), {}, outputs))
        contributions.extend(GradWrapper(target, value) for target, value in zip(target_values, outputs))
    if spec.needs[1]:
        contributions.append(GradWrapper(node.args[1], emit(ctx, "convolution.native_dw", spec.dw,
            (gradients[0], *saved), spec.metas[1])))
    return contributions


def register_rules(registry):
    if not registry.has_backward_rule(Spatial.op):
        registry.register_backward_rule(Spatial.op, vjp, multi_output=True)
