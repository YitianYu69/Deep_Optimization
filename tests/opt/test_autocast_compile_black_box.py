"""Real compiler execution must respect FP8 policy changes on a cached model.

These CPU correctness checks allow graph breaks. They do not assert full-graph
FP8 capture, compilation counts, native kernel generation, or speedups.
"""

from contextlib import nullcontext
from copy import deepcopy
import unittest

import torch
from torch import nn

from deepopt import autocast
from deepopt.opt.do_nn.functional import do_linear
from deepopt.opt.policy import fp8_plan


class AutocastCompilationBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.manual_seed(829)

    def _check_policy_changes(self, backend):
        model = nn.Linear(7, 4)
        reference = deepcopy(model)
        inputs = torch.randn(5, 7)
        upstream = torch.randn(5, 4)
        plan = fp8_plan(True)

        # Use the same in-place compilation API as configured DeepOpt runs.
        model.compile(backend=backend, fullgraph=False)
        compiled_call = model._compiled_call_impl
        self.assertIsNotNone(compiled_call)

        # Warm the dense cache before entering FP8, then revisit both policies.
        # False has the same mode class as True, while None installs no mode.
        for label, policy in (("dense", None), ("fp8", True), ("disabled", False),
                              ("dense_again", None), ("fp8_again", True)):
            with self.subTest(backend=backend, policy=label):
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                actual_x = inputs.clone().requires_grad_()
                expected_x = inputs.clone().requires_grad_()
                scope = (nullcontext() if policy is None else
                         autocast("cpu", enabled=False, fp8=policy))
                with scope:
                    actual = model(actual_x)

                expected = (do_linear(expected_x, reference.weight, reference.bias, plan=plan)
                            if policy else reference(expected_x))
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                if policy:
                    self.assertFalse(torch.equal(actual, reference(inputs)),
                                     "An FP8 scope silently reused dense execution")

                # Both backwards run after the policy context has exited.
                actual.backward(upstream)
                expected.backward(upstream)
                torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(model.bias.grad, reference.bias.grad, rtol=1e-5, atol=1e-6)
                self.assertIs(model._compiled_call_impl, compiled_call)

    def test_eager_backend_reuses_model_across_dense_fp8_and_disabled_scopes(self):
        self._check_policy_changes("eager")

    def test_aot_eager_backend_reuses_model_across_dense_fp8_and_disabled_scopes(self):
        self._check_policy_changes("aot_eager")


if __name__ == "__main__":
    unittest.main()
