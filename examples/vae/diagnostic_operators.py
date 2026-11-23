"""Checkpoint-free replay of packed multiplication's eager/native contracts.

These are isolated operator diagnostics. CPU execution checks semantic plumbing;
it does not establish Triton compilation, CUDA launches or VAE qualification.
"""

from dataclasses import replace
from math import prod, sqrt
import traceback

import torch


_CHUNK_ELEMENTS = 262144
_OPERATOR_BUDGET = {"rtol": .015, "atol": .015, "relative_l2_limit": .015}
_FULL_SE = (128, 768, 224, 224)


def _error(error):
    return {"type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error), "traceback": traceback.format_exc()}


def _tensor_metadata(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
            "stride": list(value.stride()), "elements": value.numel(),
            "bytes": value.numel() * value.element_size()}


def _chunks(left, right):
    """Copy at most one bounded logical chunk when a gradient is strided."""
    if left.numel() <= _CHUNK_ELEMENTS:
        yield left.reshape(-1), right.reshape(-1)
    elif left.shape[0] == 1:
        yield from _chunks(left[0], right[0])
    else:
        width = max(1, _CHUNK_ELEMENTS // prod(left.shape[1:]))
        for start in range(0, left.shape[0], width):
            count = min(width, left.shape[0] - start)
            yield from _chunks(left.narrow(0, start, count), right.narrow(0, start, count))


def _compare(actual, reference, *, reduced):
    budget = dict(_OPERATOR_BUDGET) if reduced else {"rtol": 0., "atol": 0., "relative_l2_limit": 0.}
    row = {"actual_dtype": str(actual.dtype), "expected_dtype": str(reference.dtype),
           "dtype_passed": actual.dtype == reference.dtype,
           "shape_passed": actual.shape == reference.shape, "budget": budget,
           "budget_scope": "predeclared_packed_operator_not_full_model_qualification"}
    if not row["shape_passed"]:
        return {**row, "passed": False, "bitwise_equal": False}
    maximum = squared = reference_squared = 0.
    finite = elementwise = True
    bitwise = row["dtype_passed"]
    nonzero_actual = nonzero_reference = 0
    for left, right in _chunks(actual, reference):
        if left.numel() == 0:
            continue
        if bitwise:
            bitwise = torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
        a, b = left.double(), right.double()
        chunk_finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        finite = finite and chunk_finite
        nonzero_actual += int(torch.count_nonzero(a))
        nonzero_reference += int(torch.count_nonzero(b))
        if chunk_finite:
            difference = a - b
            maximum = max(maximum, float(difference.abs().max()))
            squared += float(difference.square().sum())
            reference_squared += float(b.square().sum())
            elementwise = elementwise and bool(torch.isclose(
                a, b, rtol=budget["rtol"], atol=budget["atol"]).all())
    infinite_relative = reference_squared == 0 and squared != 0
    relative = sqrt(squared / reference_squared) if reference_squared else 0. if squared == 0 else None
    passed = (row["dtype_passed"] and finite and elementwise and not infinite_relative
              and relative <= budget["relative_l2_limit"])
    # Preserve the existing full-model rule as context; this diagnostic does
    # not replace its terminal-storage-dtype tolerance or promote a new one.
    epsilon = torch.finfo(reference.dtype).eps
    storage_budget = {"rtol": (2 if reference.dtype == torch.bfloat16 else 8) * epsilon,
        "atol": 2 * min(epsilon, torch.finfo(torch.float32).eps),
        "relative_l2_limit": epsilon if reference.dtype == torch.bfloat16 else 8 * epsilon}
    row.update(passed=bool(passed), finite=finite, bitwise_equal=bool(bitwise),
        max_abs=maximum if finite else None, relative_l2=relative if finite else None,
        relative_l2_infinite=infinite_relative, elementwise_passed=elementwise,
        nonzero_actual=nonzero_actual, nonzero_reference=nonzero_reference,
        full_model_storage_budget_context=storage_budget)
    return row


def _cases(scratch_bytes, full_shape):
    cases = [
        {"name": "se_both_packed_channels_last", "shape": (2, 512, 3, 5),
         "other_shape": (2, 512, 1, 1), "channels_last": True, "strided_gradient": True},
        {"name": "swiglu_both_packed_tail", "shape": (3, 1025), "other_shape": (3, 1025),
         "strided_gradient": True},
        {"name": "residual_fp32_scalar", "shape": (2, 1025), "other_shape": (),
         "other_dtype": torch.float32, "strided_gradient": True},
        {"name": "reduction_grid_stride_cancellation", "shape": (65537, 3), "other_shape": (1, 3),
         "strided_gradient": True, "cancellation": True},
        {"name": "output_tile_tail_both_packed", "shape": (2, 8193), "other_shape": (1, 8193),
         "strided_gradient": True},
        {"name": "scratch_cap_tail", "shape": (257, 1, 17), "other_shape": (1, 1, 17),
         "strided_gradient": True, "scratch_bytes": min(scratch_bytes, 128)},
    ]
    if full_shape:
        cases.append({"name": "se_full_h200_geometry", "shape": _FULL_SE,
                      "other_shape": (128, 768, 1, 1), "target_full_shape": True})
    return cases


def _native_replay(operands, gradient, descriptors, saved, *, group_size, scratch_bytes):
    """Differentiate mul IR, then bind its saves to the eager invocation's packs."""
    from deepopt._activation_memory import ActivationMemoryPlan
    from deepopt._bf16_policy import resolve_bf16_policy
    from deepopt.compiler.autodiff.activation_saving import SavedActivationPolicy
    from deepopt.compiler.autodiff.aot_auto_diff import AOTAutoDiff
    from deepopt.compiler.autodiff.packed_multiply import backward
    from deepopt.compiler.codegen.wrapper import WrapperCodegen
    from deepopt.compiler.ir import CallNode, InputNode, IRGraph, TensorMeta
    from deepopt.compiler.registry import KernelCodegenContext
    from deepopt.compiler.scheduler import Scheduler
    from deepopt.compiler.trace.builder import GraphBuilder
    from deepopt.kernel_policy import KernelPolicy

    graph = IRGraph()
    builder = GraphBuilder(graph)
    symbolic = tuple(builder.input(index, TensorMeta(tuple(value.shape), value.dtype,
        value.device, tuple(value.stride()), True), name=name) for index, (name, value)
        in enumerate(zip(("feature", "other"), operands)))
    if operands[1].ndim == 0 and operands[1].dtype == torch.float32:
        symbolic[1].value.provenance = "parameter"
    builder.output(symbolic[0] * symbolic[1])
    for node in graph.nodes:
        node.bf16_policy = resolve_bf16_policy("bf16_storage")
    trace = AOTAutoDiff(graph, saved_activation_policy=SavedActivationPolicy(True, 2, group_size)).diff()
    sources = {symbolic[index].value: descriptors[1 - index] for index in range(2)}
    bindings = {}
    for index, source in enumerate(symbolic):
        descriptor = sources[source.value]
        if descriptor[0] == "exact":
            bindings[source.value] = saved[descriptor[1]]
    for node in graph.nodes:
        if not isinstance(node, CallNode) or node.op != "codec.save_activation":
            continue
        descriptor = sources[node.args[0]]
        if descriptor[0] != "packed":
            raise ValueError("Native and eager coefficient packing decisions differ")
        components = saved[descriptor[1]:descriptor[1] + 3]
        bindings.update(zip(node.outputs(), components))
    arguments = tuple(bindings[value] for value in trace.saved_tensors)
    seed = next(node.value for node in trace.graph.nodes if isinstance(node, InputNode) and node.index == 0)
    seed.meta = replace(seed.meta, stride=tuple(gradient.stride()))
    before = [getattr(node, "op", type(node).__name__) for node in trace.graph.nodes]
    decisions = []
    patterns = backward(trace.graph, KernelCodegenContext(kernel_policy=KernelPolicy(),
        fuse_packed_multiply=True, collect_saved_state_metadata=True,
        activation_memory=ActivationMemoryPlan(multiply_backward="packed_streaming",
                                              reduction_scratch_bytes=scratch_bytes)), decisions)
    trace.graph.validate()
    artifact = WrapperCodegen(torch.nn.Module(), Scheduler(trace.graph).run(),
        enable_autodiff=False, diagnostic_phase="backward").run()
    metadata = {"owner": "DeepOpt.AOTAutoDiff", "forward_operation": "mul",
                "backward_before_fusion": before,
                "backward_after_fusion": [getattr(node, "op", type(node).__name__) for node in trace.graph.nodes],
                "selected_patterns": list(patterns), "compiler_decisions": decisions,
                "same_saved_components_bound": True, "saved_argument_count": len(arguments),
                "gradient_targets": [{"name": target.value.name, "shape": list(target.value.meta.shape),
                    "dtype": str(target.value.meta.dtype)} for target in trace.grad_targets]}
    return artifact.fn, arguments, metadata


def _oracle(gradient, descriptor, saved, target, product_dtype):
    from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation

    if descriptor[0] == "packed":
        _, offset, shape, dtype, matrix, group, bits = descriptor
        coefficient = dequantize_unpack_activation(PackedActivation(*saved[offset:offset + 3],
            matrix, group, bits=bits), dtype=dtype, row_alignment=1).reshape(shape)
    elif descriptor[0] == "exact":
        coefficient = saved[descriptor[1]]
    else:
        raise ValueError("Diagnostic requires tensor coefficients")
    return (gradient * coefficient).to(product_dtype).sum_to_size(target.shape).to(target.dtype)


def _shared(gradient, descriptor, saved, target, product_dtype, scratch_bytes, info):
    from deepopt.opt.backends._shared.packed_multiply import multiply

    if descriptor[0] != "packed":
        info.update(route="exact_coefficient", selected=False, scratch_bytes=0)
        return _oracle(gradient, descriptor, saved, target, product_dtype)
    _, offset, shape, dtype, matrix, group, bits = descriptor
    return multiply(gradient, *saved[offset:offset + 3], options={"logical_shape": shape,
        "logical_dtype": dtype, "matrix_shape": matrix, "group_size": group,
        "bits": bits, "traversal": "logical_row_major"}, shape=tuple(gradient.shape),
        product_dtype=product_dtype, target_shape=tuple(target.shape), target_dtype=target.dtype,
        scratch_bytes=scratch_bytes, execution_info=info, allow_materialized=gradient.device.type != "cuda")


def _run_case(case, row, *, device, group_size, seed, scratch_bytes):
    from deepopt.opt.nn.functional import do_mul
    from deepopt.opt.backends._shared.packed_multiply import plan_packed_multiply

    generator = torch.Generator(device=device).manual_seed(seed)
    shape, other_shape = case["shape"], case["other_shape"]
    cap = case.get("scratch_bytes", scratch_bytes)
    row["phase"] = "launch_planning"
    row["launch_plans"] = [plan_packed_multiply(shape, target, cap).as_dict() for target in (shape, other_shape)]
    row["launch_plan_scope"] = "host_metadata_not_observed_allocations"
    left = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    if case.get("channels_last"):
        left = left.contiguous(memory_format=torch.channels_last)
    right = torch.randn(other_shape, dtype=case.get("other_dtype", torch.bfloat16),
                        device=device, generator=generator)
    if case.get("strided_gradient"):
        backing = torch.randn((*reversed(shape), 2), dtype=torch.bfloat16, device=device, generator=generator)
        gradient = backing[..., 0].permute(tuple(reversed(range(len(shape)))))
    else:
        gradient = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    if case.get("cancellation"):
        index = torch.arange(prod(shape), device=device).reshape(shape)
        left.copy_((1 + (index % 7).float() / 128).to(torch.bfloat16))
        gradient.copy_(torch.where(index % 2 == 0, 1.0078125, -.99609375).to(torch.bfloat16))
    left.requires_grad_()
    right.requires_grad_()
    operands = left, right
    row.update(operands=[_tensor_metadata(value) for value in operands],
        upstream=_tensor_metadata(gradient), effective_scratch_cap_bytes=cap,
        cotangent_recipe="alternating_noninteger_BF16" if case.get("cancellation") else "seeded_normal_BF16",
        phase="eager_forward")
    eager_info = {}
    output = do_mul(*operands, plan={"fp8": False,
        "activation_compression": {"enabled": True, "group_size": group_size, "bits": 2},
        "activation_memory": {"multiply_backward": "packed_streaming", "reduction_scratch_bytes": cap}},
        _execution_info=eager_info)
    context = output.grad_fn
    descriptors, saved, product_dtype = context.descriptors, context.saved_tensors, context.product_dtype
    row["product_dtype"] = str(product_dtype)
    row["identical_cotangent_supplied_to_all_routes"] = True
    row["saved_coefficients"] = [descriptor[0] for descriptor in descriptors]
    row["saved_components"] = [_tensor_metadata(value) for value in saved]
    replay, arguments, native_metadata = _native_replay(operands, gradient, descriptors, saved,
        group_size=group_size, scratch_bytes=cap)
    row["native"] = native_metadata
    row["phase"] = "eager_backward"
    output.backward(gradient)
    eager = tuple(value.grad for value in operands)
    left.grad = right.grad = None
    del output, context
    row["eager_execution"] = eager_info
    row["phase"] = "materialized_codec_oracle"
    with torch.no_grad():
        oracle = tuple(_oracle(gradient, descriptor, saved, target, product_dtype)
                       for descriptor, target in zip(descriptors, operands))
        row["phase"] = "shared_executor"
        shared_info = [{}, {}]
        shared = tuple(_shared(gradient, descriptor, saved, target, product_dtype, cap, info)
            for descriptor, target, info in zip(descriptors, operands, shared_info))
        row["shared_execution"] = shared_info
        row["phase"] = "native_derived_replay"
        native = replay(gradient, *arguments)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row["phase"] = "comparisons"
        comparisons = {}
        for label, actual, expected in (("eager_vs_oracle", eager, oracle),
                ("shared_vs_oracle", shared, oracle), ("native_vs_oracle", native, oracle),
                ("native_vs_eager", native, eager)):
            comparisons[label] = [_compare(a, b, reduced=tuple(gradient.shape) != tuple(target.shape))
                                 for a, b, target in zip(actual, expected, operands)]
    row.update(comparisons=comparisons, passed=all(value["passed"] for rows in comparisons.values() for value in rows),
        cuda_execution_verified=device.type == "cuda",
        native_packed_cuda_execution_verified=device.type == "cuda" and any(
            decision.get("runtime_execution", {}).get("route") == "packed_streaming"
            for decision in native_metadata["compiler_decisions"]), phase="complete")
    row["status"] = "passed" if row["passed"] else "numerical_failure"


def run_packed_diagnostics(*, device="cuda:0", group_size=1024, seed=42,
                           scratch_bytes=8388608, full_shape=False):
    """Run every selected isolated replay; return scalar/metadata evidence only."""
    report = {"schema_version": 1, "kind": "packed_multiply_operator_replay", "device": str(device),
        "group_size": group_size, "bits": 2, "seed": seed, "requested_scratch_bytes": scratch_bytes,
        "full_shape_requested": full_shape, "cases": [], "passed": False, "qualified": False,
        "timing_eligible": False, "tensor_checkpoints_written": 0, "cuda_execution_verified": False,
        "predeclared_reduction_budget": dict(_OPERATOR_BUDGET),
        "budget_scope": "existing_packed_operator_budget_full_model_TensorOracle_unchanged"}
    try:
        selected = torch.device(device)
        if selected.type not in ("cpu", "cuda"):
            raise ValueError("Operator diagnostics support explicit CPU or CUDA execution")
        if group_size != 1024 or type(group_size) is not int:
            raise ValueError("Compiler compression diagnostics require group_size=1024")
        if type(scratch_bytes) is not int or scratch_bytes < 4:
            raise ValueError("scratch_bytes must be an integer >= 4")
        if type(seed) is not int or type(full_shape) is not bool:
            raise ValueError("seed must be an integer and full_shape must be a boolean")
        if selected.type == "cuda" and (not torch.cuda.is_available()
                or selected.index is not None and selected.index >= torch.cuda.device_count()):
            raise RuntimeError("Requested CUDA device is unavailable; CPU is not a CUDA fallback")
    except Exception as error:
        return {**report, "status": "unqualified", "error": _error(error)}
    report["execution_scope"] = "CUDA_operator_diagnostic" if selected.type == "cuda" else "CPU_semantic_only"
    try:
        report["environment"] = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(selected) if selected.type == "cuda" else None}
    except Exception as error:
        return {**report, "status": "unqualified", "error": _error(error)}
    for index, case in enumerate(_cases(scratch_bytes, full_shape)):
        row = {"name": case["name"], "shape": list(case["shape"]), "other_shape": list(case["other_shape"]),
               "passed": False, "cuda_execution_verified": False}
        report["cases"].append(row)
        if case.get("target_full_shape") and selected.type != "cuda":
            row.update(status="unqualified", reason="Full target SE execution requires explicit CUDA")
            continue
        try:
            from deepopt._activation_memory import activation_memory_scope
            from deepopt.kernel_policy import kernel_policy_scope

            with activation_memory_scope({"multiply_backward": "packed_streaming",
                    "reduction_scratch_bytes": case.get("scratch_bytes", scratch_bytes)}, selected.type), \
                    kernel_policy_scope("triton"), torch.autocast(selected.type, enabled=False):
                _run_case(case, row, device=selected, group_size=group_size, seed=seed + index,
                          scratch_bytes=scratch_bytes)
        except Exception as error:
            row.update(status="execution_failure", error=_error(error), passed=False)
    report["passed"] = all(case["passed"] for case in report["cases"])
    report["cuda_execution_verified"] = selected.type == "cuda" and any(
        case["cuda_execution_verified"] for case in report["cases"])
    report["status"] = ("passed" if report["passed"] else "failed") if selected.type == "cuda" else "unqualified"
    return report
