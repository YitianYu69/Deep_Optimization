"""Forward-time normalization precision snapshots shared with native AutoDiff."""

import torch


def capture_normalization_policy(x, plan=None):
    """Resolve owned plans and explicit scopes once at invocation forward.

    The shared resolver validates effective BF16 compute, FP8 and disabled
    rewriting before effects. Backward uses this frozen descriptor, never
    ambient state. A compression plan alone does not select storage policy.
    """
    from deepopt.opt.precision.bf16 import invocation_bf16_policy

    policy = invocation_bf16_policy(plan, x)
    if policy.storage_enabled:
        if plan is not None and "fp8_training" in plan:
            raise ValueError("BF16 storage normalization requires FP8 disabled")
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("BF16 storage normalization supports BF16/FP32 logical inputs")
    return policy


def normalization_output_dtype(x, policy, default, *, explicit=None):
    """Explicit dtype requests retain their meaning; affine dtypes are separate."""
    if explicit is not None:
        return explicit
    return torch.bfloat16 if policy.storage_enabled else default


def validate_normalization_backend(policy, backend, device_type):
    if policy.storage_enabled and device_type == "cuda":
        from deepopt.kernel_policy import kernel_backend

        if kernel_backend("normalization", backend=backend) != "triton":
            raise ValueError("BF16 storage normalization requires the Triton CUDA route")


def normalization_policy_receipt(policy, x, output_dtype, backend):
    return {"bf16_policy": policy.as_dict(), "input_storage_dtype": str(x.dtype),
            "output_storage_dtype": str(output_dtype), "input_gradient_storage_dtype": str(x.dtype),
            "statistics_dtype": "torch.float32", "affine_gradient_accumulation_dtype": "torch.float32",
            "storage_policy_executor": "shared_fp32_arithmetic_direct_output_store" if policy.storage_enabled else None,
            "semantic_reference_fp32_temporaries": backend == "reference",
            "cuda_execution_verified": False}
