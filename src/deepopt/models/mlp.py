"""A minimal classifier; returns logits, leaving the loss to the trainer."""

from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs):
        return self.layers(inputs)


def build_mlp(*, input_dim: int, num_classes: int, hidden_dim: int = 32, dropout: float = 0.1) -> MLP:
    """Example architecture; its settings are owned by its caller."""
    return MLP(input_dim, hidden_dim, num_classes, dropout)
