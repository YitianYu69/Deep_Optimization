"""Construct native PyTorch LR schedulers; Trainer owns step/epoch timing."""

from torch.optim import lr_scheduler as lr

from .lr_config import normalize_lr_schedule


def build_scheduler(optimizer, config):
    policy = normalize_lr_schedule(config.lr_schedule)
    if policy is None:
        return lr.ExponentialLR(optimizer, gamma=config.lr_gamma)
    kind = policy["type"]
    if kind == "constant":
        return lr.ConstantLR(optimizer, factor=1., total_iters=1)
    if kind == "step_decay":
        return lr.StepLR(optimizer, step_size=policy["step_size"], gamma=policy["gamma"])
    if kind == "exponential_decay":
        return lr.ExponentialLR(optimizer, gamma=policy["gamma"])
    if kind == "piecewise":
        return lr.MultiStepLR(optimizer, milestones=policy["milestones"], gamma=policy["gamma"])
    if kind == "linear_warmup":
        steps = policy["warmup_steps"]
        return lr.LinearLR(optimizer, start_factor=policy["start_factor"] if steps > 1 else 1.,
                           end_factor=1., total_iters=max(1, steps - 1))
    if kind == "cosine_decay":
        return lr.CosineAnnealingLR(optimizer, T_max=policy["total_steps"] - 1, eta_min=policy["eta_min"])
    warmup = policy["warmup_steps"]
    cosine = lr.CosineAnnealingLR(optimizer, T_max=policy["total_steps"] - warmup, eta_min=policy["eta_min"])
    if warmup == 1:
        return cosine
    linear = lr.LinearLR(optimizer, start_factor=policy["start_factor"], end_factor=1., total_iters=warmup - 1)
    return lr.SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup - 1])
