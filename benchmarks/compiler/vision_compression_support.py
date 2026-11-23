"""Full torchvision classifier qualification helpers; no imports allocate models."""

from copy import deepcopy
import gc

from benchmarks.compiler.inductor_reference_models import describe_reference_model


MODELS = ("resnet18", "efficientnet-b0", "vit-base")
PROFILES = ("off", "two-bit-compatible", "full-two-bit")


def compression_profile(name, group_size=1024):
    if name not in PROFILES or group_size != 1024:
        raise ValueError("Vision profiles require a known profile and group_size=1024")
    return dict(enabled=name != "off", bits=2, attention_bits=2 if name == "full-two-bit" else None,
                group_size=group_size, norm_xhat_recenter=True)


def vision_cells(models=MODELS, policies=("torch_amp", "bf16_storage"), executions=("eager", "native"), profiles=PROFILES):
    cells = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"Unsupported full vision model: {model}")
        for policy in policies:
            if policy not in ("torch_amp", "bf16_storage"):
                raise ValueError("Unknown BF16 policy")
            for execution in executions:
                if execution not in ("eager", "native"):
                    raise ValueError("Unknown vision execution")
                for profile in profiles:
                    if model != "vit-base" and profile == "two-bit-compatible" and "full-two-bit" in profiles:
                        continue
                    compression = compression_profile(profile)
                    cells.append(dict(model=model, bf16_policy=policy, execution=execution, profile=profile,
                        compression=compression, id=f"{model}-{policy}-{execution}-{profile}"))
    return cells


def construct_model(name, *, nonzero_head=False):
    import torch
    from torchvision import models

    if name not in MODELS:
        raise ValueError(name)
    model = {"resnet18": models.resnet18, "efficientnet-b0": models.efficientnet_b0,
             "vit-base": models.vit_b_16}[name](weights=None)
    expected = describe_reference_model(name)["expected_parameter_count"]
    if sum(value.numel() for value in model.parameters()) != expected:
        raise AssertionError("Original torchvision parameter count differs from the registry")
    if nonzero_head:
        if name != "vit-base":
            raise ValueError("The nonzero-head diagnostic is specific to ViT")
        with torch.no_grad():
            model.heads.head.weight.normal_(std=.02)
    return model


def prepare_model(model, policy, profile):
    from deepopt.opt.policy import prepare_fp8_model

    compression = compression_profile(profile)
    before = [(name, id(value)) for name, value in model.named_parameters()]
    config = dict(runtime=dict(device="auto", precision="bf16", bf16_policy=policy, fp8=dict(enabled=False),
                               activation_compression=compression, attention=dict(backend="native")))
    prepared, _ = prepare_fp8_model(model, False, config=config)
    if [(name, id(value)) for name, value in prepared.named_parameters()] != before:
        raise AssertionError("Preparation changed parameter identities, names or registration order")
    return prepared, config, compression


def optimizer_for(model):
    import torch

    return torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=1e-5)


def forward_invariance(name, prepared, policy, compression, images, device):
    """Same weights/RNG/kernels, value-only versus saved-state forward.

    ViT uses a separately labeled nonzero-head probe so equality cannot be
    concealed by its standard zero-initialized classifier. Main update gates
    retain the original initialization.
    """
    import torch
    from deepopt import autocast
    from examples.vae.qualification import TensorOracle, preserve_rng, rng_state, restore_rng

    with preserve_rng():
        probe = deepcopy(prepared)
        if name == "vit-base":
            with torch.no_grad():
                probe.heads.head.weight.normal_(std=.02, generator=torch.Generator().manual_seed(15121))
        # Keep preparation and arithmetic identical. An unprepared AMP Torch
        # model may select different normalization/projection kernels, so it
        # cannot isolate the effect of saving compressed activation state.
        dense = deepcopy(probe)
        state = rng_state()
        oracle = TensorOracle(16 * 2**20)
        for model, saving, sink in ((dense, False, oracle.record), (probe, True, oracle.compare)):
            restore_rng(state)
            model.to(device)
            with torch.set_grad_enabled(saving), autocast(torch.device(device).type, dtype=torch.bfloat16, fp8=False,
                    bf16_policy=policy, activation_compression=compression, config={}):
                output = model(images.to(device).detach().requires_grad_(saving))
            sink("output/forward_invariance", output.detach().cpu())
            del output
            model.cpu()
        result = oracle.finish()
    return dict(passed=True, comparison=result, nonzero_head_diagnostic=name == "vit-base",
                scope="same prepared kernels/weights/RNG: value-only no_grad versus compression-enabled saved-state forward",
                unprepared_torch_equivalence=False, lossy_gradient_quality_equivalence=False)


def emit_update(model, optimizer, images, labels, compression, policy, sink):
    import torch
    from deepopt import autocast
    from benchmarks.analysis.measurement import SavedStorageLifetimes
    from deepopt.opt.quantization.attention import observe_attention_codec
    from examples.vae.qualification import named_tree, rng_state

    optimizer.zero_grad(set_to_none=True)
    images = images.detach().requires_grad_(True)
    before = {name: value.detach().to("cpu", copy=True) for name, value in model.named_parameters()}
    tracker = SavedStorageLifetimes((*model.parameters(), *model.buffers()))
    with observe_attention_codec() as codec, torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
        with autocast(images.device.type, dtype=torch.bfloat16, fp8=False, bf16_policy=policy,
                      activation_compression=compression, config={}):
            logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    after_forward = tracker.snapshot()
    if logits.dtype != torch.bfloat16 or loss.dtype != torch.float32:
        raise AssertionError("Vision network/loss boundary differs from the BF16/FP32 contract")
    sink("output/logits", logits.detach())
    sink("output/loss", loss.detach())
    loss.backward()
    if tracker.snapshot()["live_nonpersistent_bytes"] != 0:
        raise AssertionError("Vision saved invocation storage remains after backward")
    if images.grad is None:
        raise AssertionError("Vision external input has no gradient")
    sink("input_gradient/images", images.grad)
    coverage = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Trainable parameter has no gradient: {name}")
        if parameter.dtype != torch.float32 or parameter.grad.dtype != torch.float32 or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Invalid FP32 master gradient: {name}")
        coverage[name] = float(parameter.grad.detach().float().norm())
        sink("parameter_gradient/" + name, parameter.grad.detach())
    optimizer.step()
    for name, parameter in model.named_parameters():
        sink("parameter/" + name, parameter.detach())
        sink("delta/" + name, parameter.detach() - before.pop(name).to(parameter.device))
    for name, value in named_tree(dict(model.named_buffers()), "buffer"):
        sink(name, value)
    for name, value in named_tree(optimizer.state_dict(), "optimizer"):
        sink(name, value)
    for name, value in named_tree(rng_state(), "rng"):
        sink(name, value)
    optimizer.zero_grad(set_to_none=True)
    return dict(output_dtype=str(logits.dtype), loss_dtype=str(loss.dtype), finite=True,
                gradient_norms=coverage, after_forward_saved=after_forward,
                after_backward_saved=tracker.snapshot(), attention_codec=list(codec))


def qualify_eager(name, policy, profile, *, device="cpu", batch_size=1, updates=3, max_bytes=2 * 2**30, progress=None):
    """Exercise the eager reference without depending on compiler support.

    This gate checks full-tensor finiteness, gradient coverage, codec geometry
    and saved-storage release. Native agreement is a separate paired gate.
    """
    import torch
    from deepopt._attention_policy import effective_attention_bits
    from deepopt.opt._recipe import bf16_recipe

    if updates < 3:
        raise ValueError("Full vision correctness requires at least three updates")
    torch.manual_seed(9521)
    model, _, compression = prepare_model(construct_model(name), policy, profile)
    if sum(p.numel() * p.element_size() for p in model.parameters()) >= max_bytes:
        raise MemoryError("The eager diagnostic's parameter snapshot exceeds its host bound")
    model.to(device)
    optimizer = optimizer_for(model)
    generator = torch.Generator().manual_seed(11521)
    images = torch.randn(batch_size, 3, 224, 224, generator=generator)
    labels = torch.arange(batch_size, dtype=torch.long) % 1000
    # This standalone route owns one GPU training model. Move it to CPU while
    # the independent forward probe uses the device.
    model.cpu()
    forward = forward_invariance(name, model, policy, compression, images.cpu(), device)
    model.to(device)
    images, labels = images.to(device), labels.to(device)
    def finite_sink(key, value):
        if value is not None and not torch.isfinite(value).all():
            raise AssertionError(f"Nonfinite eager state: {key}")
    evidence = []
    for update in range(updates):
        if progress is not None:
            progress(dict(update=update + 1, route="eager", phase="forward_backward_update"))
        evidence.append(emit_update(model, optimizer, images, labels, compression, policy, finite_sink))
    if name == "vit-base":
        if not any(norm > 0 for row in evidence[1:] for parameter, norm in row["gradient_norms"].items()
                   if not parameter.startswith("heads.")):
            raise AssertionError("Eager ViT backbone derivative coverage is vacuous")
        if compression["enabled"]:
            events = [event for row in evidence for event in row["attention_codec"]]
            if len(events) < 3 * 12 * updates or any(event["bits"] != effective_attention_bits(compression) for event in events):
                raise AssertionError("Eager ViT lacks the requested Q/K/V codec evidence")
    return dict(passed=True, execution="eager", model=name, parameter_count=sum(p.numel() for p in model.parameters()),
        bf16_policy=policy, bf16_recipe=bf16_recipe(policy), profile=profile, compression=compression,
        input_shape=[batch_size, 3, 224, 224], input_kind="seeded_synthetic_correctness_only", updates=evidence,
        forward_invariance=forward, lossy_training_quality="pending; finite short updates do not establish dense-training equivalence",
        numerical_scope="full-tensor finite/coverage/lifetime gate; native agreement is a separate paired gate",
        cuda_verified=str(device).startswith("cuda"), tensor_files_written=0)


def qualify_pair(name, policy, profile, *, device="cpu", batch_size=1, updates=3,
                 max_bytes=2 * 2**30, nonzero_head=False, progress=None):
    """Three paired updates, bounded host oracle, one resident GPU route."""
    import torch
    from deepopt.compiler import compilation_report
    from deepopt.config import CompileConfig
    from deepopt.engine.execution.compilation import compile_model
    from examples.vae.qualification import TensorOracle, ChunkTransport, move_optimizer, rng_state, restore_rng, saved_state_evidence
    from deepopt._attention_policy import effective_attention_bits
    from deepopt.opt._recipe import bf16_recipe

    if updates < 3:
        raise ValueError("Full vision correctness requires at least three updates")
    torch.manual_seed(9521)
    original = construct_model(name, nonzero_head=nonzero_head)
    original_count = sum(p.numel() for p in original.parameters())
    eager, config, compression = prepare_model(original, policy, profile)
    del original
    native = deepcopy(eager)
    native = compile_model(native, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
        options=dict(require_full_aot=True, use_native_autodiff=True, share_compatible_packs=True,
                     collect_saved_state_metadata=True)), config=config)
    parameters_bytes = sum(p.numel() * p.element_size() for p in eager.parameters())
    if max_bytes <= 2 * parameters_bytes:
        raise MemoryError("The bounded oracle cannot stage this full model")
    optimizers = tuple(optimizer_for(model) for model in (eager, native))
    generator = torch.Generator().manual_seed(11521)
    images = torch.randn(batch_size, 3, 224, 224, generator=generator)
    labels = torch.arange(batch_size, dtype=torch.long) % 1000
    forward = forward_invariance(name, eager, policy, compression, images, device)
    evidence = []
    for update in range(updates):
        state = rng_state()
        oracle = TensorOracle(max_bytes - parameters_bytes)
        row = dict(update=update + 1)
        for route, model, optimizer, sink in (("eager", eager, optimizers[0], oracle.record),
                                               ("native", native, optimizers[1], oracle.compare)):
            if progress is not None:
                progress(dict(update=update + 1, route=route, phase="forward_backward_update"))
            restore_rng(state)
            model.to(device)
            move_optimizer(optimizer, device)
            with ChunkTransport(sink, dict(model=name, policy=policy, profile=profile, route=route, update=update)) as transport:
                row[route] = emit_update(model, optimizer, images.to(device), labels.to(device), compression, policy, transport)
            model.cpu()
            move_optimizer(optimizer, "cpu")
            gc.collect()
        row["oracle"] = oracle.finish()
        evidence.append(row)
    report = compilation_report(native).as_dict()
    if report["runtime_fallbacks"] or not report["native_autodiff_operations"]:
        raise AssertionError("Full vision network did not use strict native AutoDiff")
    for graph in report["graphs"]:
        if graph["eager_autograd_operations"] or graph["legacy_backward_operations"]:
            raise AssertionError("Full vision network retains another backward owner")
    saved = saved_state_evidence(report, compression["enabled"])
    if name == "vit-base":
        from deepopt.opt.nn.attention import DOMultiheadAttention

        adapted = sum(isinstance(module, DOMultiheadAttention) for module in eager.modules())
        if (compression["enabled"] or policy == "bf16_storage") and adapted != 12:
            raise AssertionError("Original ViT does not own twelve prepared MHA adapters")
        if compression["enabled"]:
            events = [event for row in evidence for event in row["eager"]["attention_codec"]]
            if not events or any(event["bits"] != effective_attention_bits(compression) for event in events):
                raise AssertionError("Eager ViT Q/K/V codec evidence differs from the requested profile")
        for route in ("eager", "native"):
            if not any(norm > 0 for row in evidence[1:] for parameter, norm in row[route]["gradient_norms"].items()
                       if not parameter.startswith("heads.")):
                raise AssertionError("Zero-initialized ViT head concealed all backbone derivative coverage")
    return dict(passed=True, model=name, parameter_count=original_count, input_shape=[batch_size, 3, 224, 224],
                bf16_policy=policy, bf16_recipe=bf16_recipe(policy), profile=profile,
                compression=compression, effective_attention_bits=effective_attention_bits(compression) if name == "vit-base" else None,
                input_kind="seeded_synthetic_correctness_only", nonzero_head_diagnostic=nonzero_head,
                forward_invariance=forward, lossy_training_quality="pending; native agreement uses the matching compressed eager recipe",
                scope="network forward/native backward; eager FP32 cross-entropy and diagnostic SGD", updates=evidence,
                native_saved_state=saved, compiler_report=report, tensor_files_written=0,
                cuda_verified=str(device).startswith("cuda"))
