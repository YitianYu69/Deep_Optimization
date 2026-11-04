"""Real-model structure, shared parameter ownership, and conversion coverage."""

import unittest

import torch
from torch import fx

from deepopt.opt import transform_model
from deepopt.opt.do_nn.hf_conv1d_module import DOHFConv1D
from deepopt.opt.do_nn.linear_module import DOLinear
from deepopt.opt.do_nn.mha_module import DOMultiheadAttention
from tests.opt.full_model_support import fp8_plan, make_gpt2, make_llama, make_timm_vit, make_vit


class FullModelWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(173)

    def _transform_preserving_parameters(self, factory, mode):
        original, inputs, labels = factory()
        parameters = dict(original.named_parameters(remove_duplicate=False))
        keys = tuple(original.state_dict())
        model, report = transform_model(original, plan=fp8_plan(), trace_mode=mode)
        self.assertEqual(tuple(model.state_dict()), keys)
        actual = dict(model.named_parameters(remove_duplicate=False))
        self.assertEqual(actual.keys(), parameters.keys())
        for name, parameter in parameters.items():
            self.assertIs(actual[name], parameter, name)
        return original, model, report, inputs

    def test_vit_attention_and_mlp_projections_are_converted(self):
        _, model, report, _ = self._transform_preserving_parameters(make_vit, "full")
        self.assertIsInstance(model, fx.GraphModule)
        self.assertEqual(report.trace_mode, "full")
        modules = dict(model.named_modules())
        self.assertIsInstance(modules["encoder.layers.encoder_layer_0.self_attention"], DOMultiheadAttention)
        self.assertIsInstance(modules["encoder.layers.encoder_layer_0.mlp.0"], DOLinear)
        self.assertIsInstance(modules["encoder.layers.encoder_layer_0.mlp.3"], DOLinear)
        self.assertIsInstance(modules["heads.head"], DOLinear)

    def test_gpt2_keeps_generation_and_tied_embedding_head_with_hf_adapters(self):
        original, model, report, inputs = self._transform_preserving_parameters(make_gpt2, "modules")
        self.assertIs(model, original)
        self.assertEqual(report.trace_mode, "modules")
        self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
        self.assertIsInstance(model.lm_head, DOLinear)
        projections = [module for module in model.modules() if isinstance(module, DOHFConv1D)]
        self.assertEqual(len(projections), 4)
        model.eval()
        tokens = inputs["input_ids"][:1]
        with torch.no_grad():
            generated = model.generate(tokens, attention_mask=torch.ones_like(tokens),
                                       max_new_tokens=2, min_new_tokens=2, do_sample=False,
                                       use_cache=False)
        self.assertEqual(tuple(generated.shape), (1, tokens.shape[1] + 2))
        torch.testing.assert_close(generated[:, :tokens.shape[1]], tokens)

    def test_timm_vit_keeps_feature_api_and_converts_attention_and_mlp_projections(self):
        original, model, report, inputs = self._transform_preserving_parameters(make_timm_vit, "modules")
        self.assertIs(model, original)
        self.assertEqual(report.trace_mode, "modules")
        modules = dict(model.named_modules())
        for path in ("blocks.0.attn.qkv", "blocks.0.attn.proj", "blocks.0.mlp.fc1",
                     "blocks.0.mlp.fc2", "head"):
            self.assertIsInstance(modules[path], DOLinear, path)
        with torch.no_grad():
            features = model.forward_features(inputs["x"])
            logits = model.forward_head(features)
            expected = model(**inputs)
        self.assertEqual(tuple(features.shape), (2, 5, 32))
        torch.testing.assert_close(logits, expected)

    def test_llama_preserves_tied_embeddings_and_all_decoder_projection_parameters(self):
        original, model, report, _ = self._transform_preserving_parameters(make_llama, "auto")
        self.assertIs(model, original)
        self.assertEqual(report.trace_mode, "modules")
        self.assertIsNotNone(report.fallback_reason)
        self.assertIs(model.get_input_embeddings().weight, model.get_output_embeddings().weight)
        modules = dict(model.named_modules())
        for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                     "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            self.assertIsInstance(modules[f"model.layers.0.{path}"], DOLinear, path)
        self.assertIsInstance(model.lm_head, DOLinear)


if __name__ == "__main__":
    unittest.main()
