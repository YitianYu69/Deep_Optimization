"""Small, randomly initialized real models; no pretrained downloads."""

import importlib.util
import os
import unittest

import torch


def require_package(name):
    if importlib.util.find_spec(name) is None:
        reason = f"Optional full-model dependency {name!r} is not installed"
        if os.environ.get("DEEPOPT_REQUIRE_MODEL_TESTS") == "1":
            raise RuntimeError(reason)
        raise unittest.SkipTest(reason)


def fp8_plan(backend="reference"):
    return {
        "fp8_training": {
            "backend": backend,
            "fp8_dtypes": {
                role: torch.float8_e4m3fn for role in ("activation", "weight", "grad")
            },
        },
        "block_shapes": {role: None for role in ("activation", "weight", "grad")},
    }


def make_vit():
    require_package("torchvision")
    from torchvision.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        image_size=16, patch_size=8, num_layers=1, num_heads=4,
        hidden_dim=32, mlp_dim=64, dropout=0.0, attention_dropout=0.0,
        num_classes=7,
    )
    # torchvision starts the head at zero, which would hide all backbone gradients.
    torch.nn.init.normal_(model.heads.head.weight, std=0.02)
    inputs = {"x": torch.randn(2, 3, 16, 16)}
    return model, inputs, torch.tensor([1, 5])


def make_timm_vit():
    require_package("timm")
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        img_size=16, patch_size=8, in_chans=3, num_classes=7,
        embed_dim=32, depth=1, num_heads=4, mlp_ratio=2.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
    )
    return model, {"x": torch.randn(2, 3, 16, 16)}, torch.tensor([1, 5])


def make_gpt2():
    require_package("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=64, n_positions=32, n_embd=32, n_layer=1, n_head=4,
        n_inner=64, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        use_cache=False, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config)
    tokens = torch.randint(3, config.vocab_size, (2, 8))
    return model, {"input_ids": tokens, "use_cache": False}, tokens.clone()


def make_llama():
    require_package("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0.0,
        tie_word_embeddings=True, use_cache=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config)
    tokens = torch.randint(3, config.vocab_size, (2, 8))
    return model, {"input_ids": tokens, "use_cache": False}, tokens.clone()


def move_inputs(inputs, device):
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()}


def logits_and_loss(model, inputs, labels):
    if "input_ids" in inputs:
        output = model(**inputs, labels=labels)
        return output.logits, output.loss
    logits = model(**inputs)
    return logits, torch.nn.functional.cross_entropy(logits, labels)


def assert_relative_error(test, actual, expected, limit, description):
    actual, expected = actual.detach().float(), expected.detach().float()
    test.assertTrue(torch.isfinite(actual).all().item(), description)
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-8)
    test.assertLess(error.item(), limit, f"{description}: relative L2 error {error.item():.6f}")
