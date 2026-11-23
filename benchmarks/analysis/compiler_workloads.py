"""Resident BF16 compiler fixtures for the shared isolated analysis harness."""


def build(*, device, kind="mlp", batch=8, tokens=197, width=768, hidden=3072, output=768):
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(width, hidden), nn.Linear(hidden, output)

        def forward(self, x):
            # Deliberately retain a native activation derivative. This fixture
            # isolates Linear saved-input compression from derivative compression.
            return self.fc2(torch.ops.aten.gelu.default(self.fc1(x)))

    if kind == "linear":
        model, shape = nn.Sequential(nn.Linear(width, output)), (batch, tokens, width)
    elif kind == "mlp":
        model, shape = MLP(), (batch, tokens, width)
    elif kind == "vit-block":
        from compiler.benchmark_compiler import ViTBlock

        model, shape = ViTBlock(width, 12), (batch, tokens, width)
    elif kind == "vit-base":
        import timm

        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
        shape = (batch, 3, 224, 224)
    else:
        raise ValueError(f"Unknown BF16 compiler workload {kind!r}")
    return {"model": model.to(device), "inputs": (torch.randn(shape, device=device, requires_grad=True),),
            "benchmark_metadata": {"kind": kind, "resident_inputs": True,
                "activation_policy": "native ATen GELU; Linear compression only" if kind == "mlp" else "prepared family policy"}}
