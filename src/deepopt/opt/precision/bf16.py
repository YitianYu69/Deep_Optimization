"""Bind a prepared model's immutable storage policy to each eager forward."""

from contextlib import nullcontext

from deepopt._bf16_policy import (active_bf16_policy, bf16_policy_scope,
                                 bf16_rewrites_disabled, current_bf16_policy, resolve_bf16_policy)
from deepopt._activation_memory import (activation_memory_scope, invocation_activation_memory,
                                       resolve_activation_memory)


def invocation_bf16_policy(plan, x):
    """Resolve an operation's owned descriptor before forward/state effects.

    Legacy/default plans permit an explicit storage scope.  An owned storage
    plan rejects a conflicting explicit AMP policy, while enabled=False
    suspends rewriting without changing explicit tensor storage.
    """
    import torch
    from deepopt.opt._autocast import _get_fp8_plan

    device = x.device.type
    if device in ("cpu", "cuda") and bf16_rewrites_disabled(device):
        return resolve_bf16_policy()
    owned = resolve_bf16_policy(plan.get("bf16_policy") if isinstance(plan, dict) else None)
    if device not in ("cpu", "cuda"):
        # Meta/Fake payloads never execute storage rewrites or consult a device
        # scope; only the operation's owned descriptor is meaningful.
        return owned
    active = active_bf16_policy(device)
    if owned.storage_enabled and active is not None and active != owned:
        raise ValueError("BF16 policy differs from the owned operation plan")
    policy = owned if owned.storage_enabled else current_bf16_policy(device)
    if policy.storage_enabled:
        if _get_fp8_plan(device) not in (None, False) or isinstance(plan, dict) and "fp8_training" in plan:
            raise ValueError("bf16_storage requires FP8 disabled")
        dtype = torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else x.dtype
        if dtype != torch.bfloat16:
            raise ValueError("bf16_storage requires BF16 compute inputs or BF16 autocast")
    return policy


class _BF16PolicyForward:
    _deepopt_bf16_forward = True

    def __init__(self, owner):
        self.owner = owner
        self.override = owner.__dict__.get("forward")

    @property
    def __code__(self):
        return type(self).__call__.__code__

    def __call__(self, *args, **kwargs):
        import torch
        from torch.utils._pytree import tree_leaves
        from deepopt.opt._autocast import _linear_mode, _get_fp8_plan

        policy = resolve_bf16_policy(self.owner._deepopt_bf16_policy)
        memory = resolve_activation_memory(getattr(self.owner, "_deepopt_activation_memory_plan", None))
        from deepopt._bf16_policy import STORAGE_PRIMITIVE_RECIPE_VERSION

        if policy.storage_enabled and getattr(self.owner, "_deepopt_bf16_primitive_recipe", None) != STORAGE_PRIMITIVE_RECIPE_VERSION:
            raise ValueError("BF16 primitive recipe changed; reconstruct and recapture the model")
        tensors = tuple(value for value in tree_leaves((args, kwargs)) if isinstance(value, torch.Tensor))
        sample = next((value for value in tensors if value.is_floating_point()), None)
        if sample is None:
            sample = next(iter(tensors), None)
        if sample is None:
            sample = next(iter(self.owner.parameters()), None)
        function = self.override or type(self.owner).forward.__get__(self.owner)
        if sample is None:
            return function(*args, **kwargs)
        device = sample.device.type
        memory = invocation_activation_memory(memory, device)
        if bf16_rewrites_disabled(device):
            with activation_memory_scope(memory, device):
                return function(*args, **kwargs)
        active = active_bf16_policy(device)
        if policy.storage_enabled and active is not None and active != policy:
            raise ValueError("BF16 policy differs from the prepared model; reconstruct the original model first")
        policy = active or policy
        if policy.storage_enabled:
            if _get_fp8_plan(device) not in (None, False):
                raise ValueError("bf16_storage requires FP8 disabled")
            dtype = (torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device)
                     else sample.dtype)
            if dtype != torch.bfloat16:
                raise ValueError("A bf16_storage prepared model requires BF16 compute inputs or BF16 autocast")
        from deepopt._bf16_policy import bf16_model_region

        with activation_memory_scope(memory, device), bf16_policy_scope(policy, device), bf16_model_region(device), _linear_mode() if policy.storage_enabled else nullcontext():
            return function(*args, **kwargs)


def bind_bf16_model_policy(model, policy, *, activation_memory=None):
    policy = resolve_bf16_policy(policy)
    from deepopt._bf16_policy import STORAGE_PRIMITIVE_RECIPE_VERSION

    model._deepopt_bf16_policy = policy.as_dict()
    memory = resolve_activation_memory(activation_memory)
    model._deepopt_activation_memory_plan = memory
    if policy.storage_enabled:
        previous = getattr(model, "_deepopt_bf16_primitive_recipe", STORAGE_PRIMITIVE_RECIPE_VERSION)
        if previous != STORAGE_PRIMITIVE_RECIPE_VERSION:
            raise ValueError("BF16 primitive recipe differs from the prepared model")
        model._deepopt_bf16_primitive_recipe = STORAGE_PRIMITIVE_RECIPE_VERSION
    # A compiled precision body already has its policy facade outside Dynamo.
    # Re-preparation must preserve that body: Module.compile reads forward
    # lazily, so wrapping it again would trace the host ContextVar mutations.
    # AMP default models retain their existing forward/capture contract.
    if ((policy.storage_enabled or not memory.is_default)
            and not getattr(model.forward, "_deepopt_bf16_forward", False)
            and not getattr(model.forward, "_deepopt_compiler_precision_body", False)):
        model.forward = _BF16PolicyForward(model)
    return model
