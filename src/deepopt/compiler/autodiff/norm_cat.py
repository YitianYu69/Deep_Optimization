"""Default-off body GroupNorm destination writes for a following BF16 cat.

Only the body producer is replaced: skip keeps its original allocation for
later residual consumers. The destination and its body view never enter a
cache or escape separately. Native AutoDiff is derived before this pass, so
the original normalization residual leaves and backward rules stay unchanged.
"""

from math import prod

import torch

from ..ir import CallNode, GetAttrNode, InputNode, TensorValue
from .fusion_contracts import candidate, select
from .semantic import recorded
from .spatial import Normalization


PATTERN = "group_norm_cat_body_destination"


def _contiguous(meta):
    if meta is None or meta.stride is None:
        return False
    expected = 1
    for size, stride in zip(reversed(meta.shape), reversed(meta.stride)):
        if size > 1 and stride != expected:
            return False
        expected *= max(size, 1)
    return True


class _NormCatBody:
    """One invocation-local allocation, direct body writes and one skip copy."""

    def __init__(self, normalization, skip_meta, output_meta):
        self.normalization = normalization
        self.skip_shape = tuple(skip_meta.shape)
        self.output_shape = tuple(output_meta.shape)
        self.body_shape = tuple(normalization.geometry.shape)
        self.skip_channels = self.skip_shape[1]
        self.forward_steps = ("allocate_cat_destination", "normalization_body_destination", "copy_skip_half")
        self.source = "GroupNorm body writes into BF16 cat destination; skip remains separate (CUDA unqualified)"

    def __getattr__(self, name):
        return getattr(self.normalization, name)

    @torch.no_grad()
    def forward(self, x, weight, bias, running_mean, running_var, counter, skip):
        from deepopt.opt.backends._shared.normalization import normalization_kernel_scope

        spec = self.normalization
        if (tuple(x.shape) != self.body_shape or tuple(skip.shape) != self.skip_shape
                or x.dtype != torch.bfloat16 or skip.dtype != torch.bfloat16
                or skip.device != x.device or not x.is_contiguous() or not skip.is_contiguous()):
            raise ValueError("GroupNorm cat body inputs violate the captured dense BF16 metadata")
        if (getattr(spec, "native_group", None) is not None or spec.kind != "group"
                or spec.output_layout is not None or spec.policy.enabled
                or spec.output_dtype != torch.bfloat16
                or spec.normalization_variant not in ("auto", "baseline")):
            raise ValueError("GroupNorm cat body executor no longer satisfies its output contract")
        output = torch.empty(self.output_shape, device=x.device, dtype=torch.bfloat16)
        body = output.narrow(1, self.skip_channels, self.body_shape[1])
        with normalization_kernel_scope(spec.normalization_variant, row_chunk=spec.normalization_tuning[0],
                                        max_programs=spec.normalization_tuning[1]):
            values = spec.forward(x, weight, bias, running_mean, running_var, counter, output=body)
        if not isinstance(values, tuple) or len(values) != 1 + len(spec.state_names) or values[0] is not body:
            raise RuntimeError("GroupNorm destination executor did not return its declared body view and saved slots")
        output.narrow(1, 0, self.skip_channels).copy_(skip)
        spec.module.last_execution.update(norm_cat_body_destination=True,
            norm_cat_skip_copy_bytes=skip.numel() * skip.element_size(),
            norm_cat_body_materialization_bytes=0)
        return (output, *values[1:])


def _reason(graph, norm, cat, spec, skip, context):
    body, output = norm.values[0], cat.values[0]
    if type(spec) is not Normalization:
        return "normalization_activation_epilogue"
    if (spec.kind != "group" or getattr(spec, "native_group", None) is not None
            or spec.output_layout is not None or spec.policy.enabled
            or "fp8_training" in spec.plan):
        return "unsupported_normalization_executor"
    if not spec.bf16_policy.storage_enabled or spec.output_dtype != torch.bfloat16:
        return "bf16_storage_contract"
    if norm.bf16_policy != spec.bf16_policy or cat.bf16_policy != spec.bf16_policy:
        return "policy_contract_mismatch"
    if spec.normalization_variant not in ("auto", "baseline"):
        return "normalization_variant"
    if spec.empty or spec.public_count != 1 or len(norm.values) != 1 + len(spec.state_names):
        return "normalization_saved_slot_contract"
    if norm.kwargs or len(norm.args) != 6 or any(value is not None for value in norm.args[3:]):
        return "normalization_arguments"
    source = norm.args[0]
    if not isinstance(source, TensorValue):
        return "normalization_source"
    metas = (source.meta, skip.meta, body.meta, output.meta)
    if any(meta is None or meta.device is None or len(meta.shape) != 4 or any(size <= 0 for size in meta.shape)
           or meta.representation != "dense" or meta.storage_dtype is not None
           or meta.dtype != torch.bfloat16 for meta in metas):
        return "dense_bf16_nchw_required"
    if any(torch.device(meta.device).type != "cuda" for meta in metas):
        return "device"
    if len({str(torch.device(meta.device)) for meta in metas}) != 1:
        return "device_mismatch"
    if not all(_contiguous(meta) for meta in metas):
        return "contiguous_layout_required"
    if (source.meta.shape != body.meta.shape or body.meta.shape != tuple(spec.geometry.shape)
            or skip.meta.shape[0] != body.meta.shape[0] or skip.meta.shape[2:] != body.meta.shape[2:]
            or output.meta.shape != (body.meta.shape[0], skip.meta.shape[1] + body.meta.shape[1], *body.meta.shape[2:])):
        return "cat_geometry"
    if body.users != [cat]:
        return "body_output_fanout_or_saved_lease"
    before, after = graph.nodes.index(norm), graph.nodes.index(cat)
    if before >= after or any(not isinstance(node, (InputNode, GetAttrNode)) for node in graph.nodes[before + 1:after]):
        return "intervening_operation"
    if skip.producer not in graph.nodes or graph.nodes.index(skip.producer) >= before:
        return "skip_dependency_not_available"
    policy = spec.kernel_policy
    if policy.route("normalization", "normalization.forward") != "triton" or spec.backend not in ("triton", "mixed", "inherit"):
        return "normalization_backend_has_no_destination"
    if (spec.compression["enabled"] and policy.pinned("activation_codec", "normalization.forward")
            and policy.route("activation_codec", "normalization.forward") != "triton"):
        return "consumed_role_backend_conflict"
    return None


def forward(graph, context):
    """Return serializable candidate decisions and selected pattern names."""
    from .optimization import rebuild

    candidates, patterns = [], []
    enabled = bool(getattr(context, "fuse_norm_cat_body", False)
                   and "forward" in getattr(context, "fusion_phases", ("forward", "backward")))
    for cat in list(graph.nodes):
        if (not isinstance(cat, CallNode) or cat.op != "autodiff_storage_cat"
                or len(cat.values) != 1 or cat.kwargs or len(cat.args) != 2
                or type(cat.args[1]) is not int or cat.args[1] != 1
                or not isinstance(cat.args[0], (tuple, list)) or len(cat.args[0]) != 2):
            continue
        skip, body = cat.args[0]
        if not isinstance(skip, TensorValue) or not isinstance(body, TensorValue):
            continue
        norm = body.producer
        if not isinstance(norm, CallNode) or not norm.values or norm.values[0] is not body:
            continue
        spec = norm.specification
        base = getattr(spec, "normalization", spec)
        if type(base) is not Normalization:
            continue
        record = candidate(graph, cat, base, "forward", PATTERN, requested=enabled)
        record.update(transformation="destination_passing", kernel_fusion=False,
                      qualification="experimental_cuda_unqualified",
                      implementation="triton_group_norm_body_destination_with_skip_copy",
                      phase="train" if base.module.training else "eval")
        candidates.append(record)
        if not enabled:
            continue
        reason = _reason(graph, norm, cat, spec, skip, context)
        if reason:
            record.update(reason=reason, eligible=False)
            continue
        wrapper = _NormCatBody(spec, skip.meta, cat.values[0].meta)
        replacement = CallNode("normalization.cat_body_forward", recorded(wrapper, wrapper.forward, "forward"),
            (*norm.args, skip), {}, (cat.values[0], *norm.values[1:]), wrapper)
        replacement.bf16_policy = norm.bf16_policy
        for name in ("module_path", "source_node", "activation_memory"):
            if hasattr(norm, name):
                setattr(replacement, name, getattr(norm, name))
        select(replacement, record, roles={"normalization_body_destination": ["_apply"]},
            consumed=(norm.op, cat.op), source_files=("compiler/autodiff/norm_cat.py",
                "compiler/autodiff/spatial.py",
                "opt/nn/autograd/normalization/function.py", "opt/backends/_shared/normalization_output.py",
                "opt/backends/triton/normalization/__init__.py", "opt/backends/triton/normalization/forward.py"))
        record["replacement_contract"].update(destination_ownership="fresh_invocation_local",
            output_layout="contiguous_cat_with_nonoverlapping_channel_slices", skip_storage="original_separate",
            body_output_materialization_bytes=0, skip_copy_bytes=prod(skip.meta.shape) * skip.meta.dtype.itemsize,
            saved_slot_names=tuple(value.name for value in norm.values[1:]),
            saved_slot_contract="unchanged_original_normalization_leaves", peak_saving_bytes=None)
        record["kernel_evidence"].update(required_native_operations=("aten::copy_",),
            forbidden_native_operations=("aten::cat",), body_destination_store=True)
        graph.nodes[graph.nodes.index(norm)] = replacement
        graph.nodes.remove(cat)
        rebuild(graph)
        patterns.append(PATTERN)
    return candidates, tuple(patterns)
