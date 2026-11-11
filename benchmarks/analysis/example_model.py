"""Executable factory contract: a small classifier with optional pinned input transfer.

Replace build() with your own model/loader/loss, keeping the return contract.
No pretrained weights, datasets, downloads or additional dependencies are used.
"""


def build(device, batch_size=2, sequence_length=16, width=64, heads=4, layers=2,
          classes=10, input_mode="resident", batches=8):
    import torch
    from torch import nn
    from torch.nn import functional as F

    if input_mode not in ("resident", "pinned"):
        raise ValueError("input_mode must be resident or pinned")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

        def forward(self, x):
            h = self.norm1(x)
            x = x + self.attention(h, h, h, need_weights=False)[0]
            return x + self.mlp(self.norm2(x))

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(width, width)
            self.blocks = nn.Sequential(*(Block() for _ in range(layers)))
            self.head = nn.Linear(width, classes)

        def forward(self, x):
            return self.head(self.blocks(self.embedding(x)).mean(1))

    model = Classifier().to(device)
    samples = [(torch.randn(batch_size, sequence_length, width, dtype=torch.bfloat16),
                torch.randint(classes, (batch_size,))) for _ in range(batches)]
    if input_mode == "pinned" and device.type == "cuda":
        samples = [(x.pin_memory(), y.pin_memory()) for x, y in samples]
    state = {"index": 0, "target": samples[0][1].to(device)}
    resident = samples[0][0].to(device)

    def reset():
        state["index"] = 0
        state["target"] = samples[0][1].to(device)

    def fetch():
        x, y = samples[state["index"] % len(samples)]
        state["index"] += 1
        state["target"] = y.to(device, non_blocking=True)
        return (x.to(device, non_blocking=True),), {}

    result = {"model": model, "inputs": (resident,), "kwargs": {},
              "loss_fn": lambda output: F.cross_entropy(output.float(), state["target"]),
              "fingerprint_data": samples}
    if input_mode == "pinned":
        result.update(fetch=fetch, reset=reset)
    return result
