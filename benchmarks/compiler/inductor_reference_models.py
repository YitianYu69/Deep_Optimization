"""Download-free full training models for the local Inductor reference corpus.

Importing this module is standard-library only. No compiler, optimizer update,
tensor persistence, dependency installation or pretrained download occurs here.
The runner owns compilation and bounded same-recipe eager validation.
"""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from importlib import import_module
import math
import random


MODEL_NAMES = ("resnet18", "vit-base", "efficientnet-b0", "vae-v2")
RECIPES = ("torch-bf16",)
_ALIASES = {"vit_b_16": "vit-base", "efficientnet_b0": "efficientnet-b0", "vae_v2": "vae-v2", "vae": "vae-v2"}
_SPECS = {
    "resnet18": {"constructor": "torchvision.models.resnet18", "parameters": 11689512, "kind": "classifier"},
    "vit-base": {"constructor": "torchvision.models.vit_b_16", "parameters": 86567656, "kind": "classifier"},
    "efficientnet-b0": {"constructor": "torchvision.models.efficientnet_b0", "parameters": 5288548, "kind": "classifier"},
    "vae-v2": {"constructor": "examples.vae.mbconv_vae.MBConvVAE_V2", "parameters": 32317999, "kind": "vae"},
}
VAE_MODEL_OPTIONS = {"base_channels": 32, "max_channels": 512, "blocks_per_stage": 2, "latent_channels": 4}


@contextmanager
def observe_stochastic_branches(model):
    """Untimed eager-only observations; never install hooks on compiled models.

    Count a dropped branch only where a nonzero input makes the mask observable.
    Zero inputs are explicitly unknown. No RNG changes or tensor copies occur.
    """
    observations, handles = [], []

    def hook(name):
        def record(module, inputs, output):
            value = inputs[0].detach()
            result = output.detach()
            units = value.shape[0] if module.mode == "row" else 1
            incoming = value.reshape(units, -1).count_nonzero(dim=1) > 0
            outgoing = result.reshape(units, -1).count_nonzero(dim=1) > 0
            observations.append({"module": name, "p": module.p, "mode": module.mode,
                "training": module.training, "units": units,
                "observed_dropped": int((incoming & ~outgoing).sum()),
                "observed_kept": int((incoming & outgoing).sum()),
                "unobservable_zero_inputs": int((~incoming).sum()),
                "scope": "untimed eager native Torch branch output; no compiled hook or RNG perturbation"})
        return record

    try:
        for name, module in model.named_modules():
            cls = type(module)
            if cls.__name__ == "StochasticDepth" and cls.__module__.startswith("torchvision."):
                handles.append(module.register_forward_hook(hook(name)))
        yield observations
    finally:
        for handle in handles:
            handle.remove()


def describe_reference_model(name, recipe="torch-bf16", *, batch_size=1, image_size=224,
                             seed=42, beta=1e-6, group_size=1024, probe_label=None):
    """Return a JSON-safe plan without importing Torch, torchvision or DeepOpt."""
    canonical = _ALIASES.get(name, name)
    if canonical not in _SPECS:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}; choose from {RECIPES}")
    if type(batch_size) is not int or batch_size < 1 or type(image_size) is not int or image_size < 1:
        raise ValueError("batch_size and image_size must be positive integers")
    if type(seed) is not int or type(beta) not in (float, int) or not math.isfinite(beta) or beta < 0:
        raise ValueError("seed must be an integer and beta must be finite and nonnegative")
    if group_size != 1024 or type(group_size) is not int:
        raise ValueError("Reference compression testing requires group_size=1024")
    if probe_label is not None and (not isinstance(probe_label, str) or not probe_label.strip()):
        raise ValueError("An external probe requires a nonempty explicit probe_label")
    if probe_label is None and image_size != 224:
        raise ValueError("Full reference models require image_size=224; label other geometries as diagnostic probes")
    specification = _SPECS[canonical]
    return {"name": canonical, "recipe": recipe, "constructor": specification["constructor"],
            "model_kind": specification["kind"], "weights": None, "pretrained": False,
            "architecture": dict(VAE_MODEL_OPTIONS) if canonical == "vae-v2" else {"num_classes": 1000},
            "expected_parameter_count": specification["parameters"],
            "input_shape": [batch_size, 3, image_size, image_size], "seed": seed, "beta": beta,
            "model_scope": "diagnostic_probe" if probe_label is not None else "full_model",
            "probe_label": probe_label, "full_model_reference": probe_label is None,
            "precision": {"autocast": "torch.bfloat16", "master_parameters": "torch.float32",
                          "loss_accumulation": "torch.float32"},
            "compression": {"enabled": False, "group_size": 1024},
            "operator_policy": "native_torch_ops_only",
            "native_fx_provenance_required": True,
            "compiler_backend": "inductor", "autodiff_owner": "pytorch_aotautograd",
            "autodiff_owner_scope": "requested_compiler_pipeline_not_execution_proof",
            "eager_autodiff_owner": "pytorch_autograd", "adapter_executes_compiler": False,
            "deepopt_native_autodiff": False, "optimizer_scope": "diagnostic_sgd_not_original_recipe",
            "original_vae_optimizer_used": False, "original_vae_training_qualification": False,
            "optimizer": {"class": "torch.optim.SGD", "lr": .001, "momentum": .9, "weight_decay": 1e-5},
            "cuda_verified": False, "download_policy": "weights_None_no_downloads",
            "input_policy": "seeded_synthetic_not_dataset_quality_evidence"}


def capture_rng():
    """Capture process RNG state in RAM; this does not write tensor files."""
    import torch

    numpy = _numpy_module()
    return {"torch": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": deepcopy(numpy.random.get_state()) if numpy is not None else None}


def restore_rng(state):
    import torch

    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    if state["numpy"] is not None:
        numpy = _numpy_module()
        if numpy is None:
            raise RuntimeError("Cannot restore captured NumPy RNG without NumPy")
        numpy.random.set_state(state["numpy"])


def reset_rng(seed):
    import torch

    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy = _numpy_module()
    if numpy is not None:
        numpy.random.seed(seed % 2**32)


def _numpy_module():
    try:
        return import_module("numpy")
    except ImportError:
        return None


def optimizer_factory(parameters):
    """Diagnostic optimizer only, including for VAE; never its private optimizer."""
    import torch

    return torch.optim.SGD(parameters, lr=.001, momentum=.9, weight_decay=1e-5)


def loss_fn(outputs):
    """Loss selector for the model wrapper's named outputs."""
    return outputs["loss"]


def _installed_identity():
    # This helper enforces non-editable installed-package use before DeepOpt is
    # imported. The raw torchvision routes do not need DeepOpt at all.
    if __package__:
        return import_module(f"{__package__}.installed_runtime").installed_identity()
    return import_module("installed_runtime").installed_identity()


def _full_network(metadata):
    if metadata["model_kind"] == "vae":
        factory = import_module("examples.vae.mbconv_vae").MBConvVAE_V2
        return factory(**VAE_MODEL_OPTIONS)
    try:
        models = import_module("torchvision.models")
    except (ImportError, RuntimeError, OSError) as error:
        raise RuntimeError("Full classifier references require a compatible local torchvision installation; no model substitute or download is used") from error
    factory = getattr(models, metadata["constructor"].rsplit(".", 1)[1])
    return factory(weights=None)


def _training_wrapper(network, kind, beta, device_type):
    import torch
    from torch.nn import functional as F

    class TrainingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = network

        def forward(self, images, targets=None):
            with torch.autocast(device_type, dtype=torch.bfloat16):
                result = self.network(images)
            with torch.autocast(device_type, enabled=False):
                if kind == "classifier":
                    loss = F.cross_entropy(result.float(), targets)
                    return {"loss": loss, "logits": result}
                reconstruction, mu, logvar = result
                l1 = F.l1_loss(reconstruction.float(), images.float())
                kl = .5 * (mu.float().square() + logvar.float().exp() - 1 - logvar.float()).sum((1, 2, 3)).mean()
                return {"loss": l1 + beta * kl, "reconstruction": reconstruction,
                        "mu": mu, "logvar": logvar, "l1": l1, "kl": kl}

    return TrainingModel()


def build_reference_model(name, recipe="torch-bf16", device="cuda:0", batch_size=1,
                          image_size=224, seed=42, beta=1e-6, group_size=1024, *,
                          probe_factory=None, probe_label=None, input_requires_grad=True):
    """Build one uncompiled full model, its loss, inputs and diagnostic optimizer.

    ``probe_factory`` takes no arguments, must have an explicit label and never
    produces full-model evidence. ``execution_context`` is a no-op compatibility
    hook; BF16 Torch autocast and the loss are inside the training wrapper.
    Factories receive no pretrained weights. Missing dependencies fail closed.
    """
    if probe_factory is not None and probe_label is None:
        raise ValueError("An external probe_factory requires an explicit probe_label")
    metadata = describe_reference_model(name, recipe, batch_size=batch_size, image_size=image_size,
        seed=seed, beta=beta, group_size=group_size, probe_label=probe_label)
    import torch

    selected_device = torch.device(device)
    if selected_device.type not in ("cpu", "cuda"):
        raise ValueError("Reference execution device must be cpu or cuda")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reference execution requested on a host without CUDA")
    if selected_device.type == "cuda":
        with torch.cuda.device(selected_device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The selected CUDA device does not support BF16")
    installed = None
    if metadata["model_kind"] == "vae" and probe_factory is None:
        installed = _installed_identity()
    reset_rng(seed)
    network = probe_factory() if probe_factory is not None else _full_network(metadata)
    if not isinstance(network, torch.nn.Module):
        raise TypeError("Model factory must return torch.nn.Module")
    if any(parameter.dtype != torch.float32 for parameter in network.parameters()):
        raise ValueError("Reference model must start with FP32 master parameters")
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    network_class = f"{type(network).__module__}.{type(network).__qualname__}"
    if probe_label is None and parameter_count != metadata["expected_parameter_count"]:
        raise ValueError(f"Full {metadata['name']} identity differs: {parameter_count} parameters, expected {metadata['expected_parameter_count']}")
    model = _training_wrapper(network, metadata["model_kind"], beta, selected_device.type).to(selected_device).train()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = metadata["input_shape"]
    images = torch.randn(shape, generator=generator) if metadata["model_kind"] == "classifier" else torch.rand(shape, generator=generator) * 2 - 1
    images = images.to(selected_device).requires_grad_(input_requires_grad)
    if metadata["model_kind"] == "classifier":
        targets = torch.randint(0, 1000, (batch_size,), generator=generator, dtype=torch.int64).to(selected_device)
        inputs = (images, targets)
    else:
        inputs = (images,)
    metadata.update(parameter_count=parameter_count,
                    trainable_parameter_count=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                    parameter_bytes=sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
                    buffer_bytes=sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()),
                    network_class=network_class, device=str(selected_device), torch_version=str(torch.__version__),
                    execution_profile="cpu_diagnostic" if selected_device.type == "cpu" else "cuda_requested_not_verified",
                    installed_deepopt=installed, input_requires_grad=input_requires_grad,
                    forward_context_required=False,
                    prepared_custom_autograd=False,
                    vae_blur_pool_policy="existing_transparent_primitive_decomposition" if metadata["model_kind"] == "vae" else None)
    return {"model": model, "inputs": inputs, "loss_fn": loss_fn,
            "optimizer": optimizer_factory(model.parameters()), "optimizer_factory": optimizer_factory,
            "capture_rng": capture_rng, "restore_rng": restore_rng, "reset_rng": reset_rng,
            "execution_context": nullcontext, "metadata": metadata}


def build_model(name, device="cuda:0", recipe="torch-bf16", **kwargs):
    """Compatibility entry point for runners using device as the second argument."""
    return build_reference_model(name, recipe=recipe, device=device, **kwargs)
