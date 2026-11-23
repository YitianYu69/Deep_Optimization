"""A custom operation's declared schema participates in exact resume identity."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

import deepopt
from deepopt import register_operation
from deepopt.engine.checkpointing.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from deepopt.operations import operation_recipes
from deepopt.opt._recipe import kernel_recipe


def _saved_state(ctx, inputs, output):
    if ctx.needs_grad(inputs["x"]):
        ctx.save("weight", inputs["weight"])
    if ctx.needs_grad(inputs["weight"]):
        ctx.save("source", inputs["x"])


def _vjp(state, gradient):
    return {**({"x": gradient * state["weight"]} if "weight" in state else {}),
            **({"weight": gradient * state["source"]} if "source" in state else {})}


def operation(schema):
    @register_operation(name="checkpoint_custom", schema=schema, state_operands=("weight",),
        fake=lambda x, weight: torch.empty_like(x), saved_state=_saved_state, vjp=_vjp)
    def multiply(x, weight):
        return x * weight
    return multiply


class _Model(nn.Module):
    def __init__(self, custom):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.5))
        self.custom = custom

    def encode(self, x):
        return self.custom(x, self.weight)

    def forward(self, x):
        return self.encode(x)


class RegisteredOperationRecipeTests(unittest.TestCase):
    def tearDown(self):
        deepopt.reset_config()

    def test_schema_discovery_precedes_forward_and_survives_compiler_wrappers(self):
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model

        model = _Model(operation("one"))
        expected = ({"name": "checkpoint_custom", "schema": "one", "route": "explicit_vjp",
                     "state_operands": ("weight",)},)
        self.assertEqual(operation_recipes(model), expected)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                           options={"require_full_aot": True}))
        self.assertEqual(operation_recipes(model), expected)

    def test_decomposition_helpers_and_explicit_dynamic_declarations(self):
        @register_operation(name="checkpoint_composite", decomposition=True, schema="three")
        def composite(x):
            return x + x

        def helper(x):
            return composite(x)

        class Composite(nn.Module):
            def forward(self, x):
                return helper(x)

        model = Composite()
        self.assertEqual(operation_recipes(model), ({"name": "checkpoint_composite", "schema": "three",
            "route": "decomposition", "state_operands": ()},))
        # A dispatch table outside the statically referenced method can instead
        # declare its possible operations without a first execution.
        plain = nn.Linear(1, 1)
        self.assertIsNone(kernel_recipe(plain))
        plain.__deepopt_operations__ = (operation("dynamic"),)
        self.assertEqual(operation_recipes(plain)[0]["schema"], "dynamic")

    def test_exact_resume_rejects_changed_schema_before_loading_any_state(self):
        def trainer(schema):
            model = _Model(operation(schema))
            return deepopt.Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                device="cpu", loss_fn=nn.MSELoss(), config={"runtime": {
                    "activation_compression": {"enabled": True, "bits": 2, "group_size": 16}},
                    "logging": {"level": "ERROR"}})

        source, same, changed = trainer("one"), trainer("one"), trainer("two")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            # No model forward has run: contract discovery must already work.
            save_checkpoint(path, source, generators={}, history=[])
            payload = read_checkpoint(path)
        self.assertEqual(payload["trainer_settings"]["kernel_recipe"]["registered_operations"][0]["schema"], "one")
        restore_checkpoint(payload, same, generators={}, batch_size_mode="manual")
        with torch.no_grad():
            changed.model.weight.fill_(7)
        before = deepcopy(changed.model.state_dict())
        with self.assertRaisesRegex(ValueError, "kernel policy or compiler versions differ"):
            restore_checkpoint(payload, changed, generators={}, batch_size_mode="manual")
        for name, value in changed.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
