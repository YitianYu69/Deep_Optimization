"""Small, pointer-free multi-output ABI for native training composites.

The descriptor documents existing saved TensorValues. It grants no allocation
reuse, changes no derivative rule and is independent of a runtime invocation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingFusionABI:
    name: str
    output_roles: tuple[str, ...]
    rounding_boundaries: tuple[str, ...]
    iteration_layout: str = "contiguous_nchw"
    index_bits: int = 64

    def __post_init__(self):
        if (not self.name or not self.output_roles or len(set(self.output_roles)) != len(self.output_roles)
                or self.index_bits != 64):
            raise ValueError("Training fusion requires named unique outputs and 64-bit indexed addressing")

    def receipt(self, values):
        if len(values) != len(self.output_roles) or any(value.meta is None for value in values):
            raise ValueError("Training fusion output metadata does not match its declared ABI")
        return {"name": self.name, "outputs": tuple({"role": role, "name": value.name,
                "shape": tuple(value.meta.shape), "dtype": str(value.meta.dtype),
                "stride": value.meta.stride, "device": str(value.meta.device),
                "representation": value.meta.representation}
                for role, value in zip(self.output_roles, values)),
            "rounding_boundaries": self.rounding_boundaries, "iteration_layout": self.iteration_layout,
            "index_bits": self.index_bits, "reuse_permission": False,
            "autodiff_owner": "deepopt_native"}

    def check_runtime(self, tensors, metadata):
        if len(tensors) != len(self.output_roles) or len(metadata) != len(tensors):
            raise ValueError("Training fusion runtime output arity differs from its declared ABI")
        import torch

        for role, tensor, meta in zip(self.output_roles, tensors, metadata):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != meta.shape
                    or tensor.dtype != meta.dtype or tensor.device.type != torch.device(meta.device).type
                    or torch.device(meta.device).index is not None and tensor.device.index != torch.device(meta.device).index
                    or meta.stride is not None and tuple(tensor.stride()) != meta.stride):
                raise ValueError(f"Training fusion runtime metadata mismatch for {role}")
        return tuple(tensors)
