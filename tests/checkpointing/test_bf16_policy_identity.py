"""In-memory exact-resume identity and CPU replay-policy simulations."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from deepopt import Trainer
from deepopt._bf16_policy import BF16Policy, bf16_policy_scope
from deepopt.config import reset_config
from deepopt.cuda_graph._validation import operation_policy_spec, tensor_spec
from deepopt.cuda_graph.capture import CapturedFunction
from deepopt.cuda_graph.model import _policy_description
from deepopt.engine.checkpointing.checkpoint import _checkpoint_payload, restore_checkpoint
from deepopt.engine.checkpointing.identity import compatibility_config, model_bf16_policy, normalized_bf16_settings
from deepopt.tasks.regression import RegressionTask


class BF16IdentityTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def trainer(self):
        model = torch.nn.Linear(2, 1)
        return Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                       task=RegressionTask(), device="cpu", config={})

    def test_legacy_missing_is_only_amp_and_invalid_version_fails_closed(self):
        self.assertEqual(compatibility_config({}), compatibility_config({"runtime": {"bf16_policy": "torch_amp"}}))
        legacy = {"runtime": {"device": "cpu"}}
        for value in ("torch_amp", {"name": "torch_amp", "version": 1}):
            self.assertEqual(compatibility_config(legacy), compatibility_config(
                {"runtime": {"device": "cpu", "bf16_policy": value}}))
        self.assertNotEqual(compatibility_config(legacy), compatibility_config(
            {"runtime": {"device": "cpu", "bf16_policy": "bf16_storage"}}))
        with self.assertRaises(ValueError):
            normalized_bf16_settings({"bf16_policy": {"name": "torch_amp", "version": 2}})
        self.assertEqual(legacy, {"runtime": {"device": "cpu"}})

    def test_owned_descriptor_survives_ambient_change_and_inconsistent_plan_rejected(self):
        model = torch.nn.Linear(2, 1)
        model._deepopt_bf16_policy = BF16Policy("bf16_storage").as_dict()
        with bf16_policy_scope("torch_amp", "cpu"):
            self.assertEqual(model_bf16_policy(model, {}), BF16Policy("bf16_storage"))
        model.plan = {"bf16_policy": BF16Policy().as_dict()}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            model_bf16_policy(model, {})

    def test_nested_owned_policy_is_preserved_and_wrapper_conflicts_reject_in_ram(self):
        child = torch.nn.ReLU()
        child._deepopt_bf16_policy = BF16Policy("bf16_storage").as_dict()
        wrapped = torch.nn.Sequential(child)
        self.assertEqual(model_bf16_policy(wrapped, {"runtime": {"bf16_policy": "torch_amp"}}),
                         BF16Policy("bf16_storage"))
        wrapped._deepopt_bf16_policy = BF16Policy().as_dict()
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            model_bf16_policy(wrapped, {})
        wrapped._deepopt_bf16_policy = BF16Policy("bf16_storage").as_dict()
        child.plan = {"bf16_policy": BF16Policy().as_dict()}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            model_bf16_policy(wrapped, {})

    def test_legacy_payload_resumes_in_ram_and_changed_policy_rejects_before_load(self):
        source = self.trainer()
        payload = deepcopy(_checkpoint_payload(source, source.config, generators={}, history=[]))
        payload["config"]["runtime"].pop("bf16_policy", None)
        payload["trainer_settings"].pop("bf16_policy", None)
        target = self.trainer()
        restore_checkpoint(payload, target, {}, generators={})
        changed = deepcopy(payload)
        changed["trainer_settings"]["bf16_policy"] = BF16Policy("bf16_storage").as_dict()
        with patch.object(target.model, "load_state_dict") as load:
            with self.assertRaisesRegex(ValueError, "trainer settings"):
                restore_checkpoint(changed, target, {}, generators={})
            load.assert_not_called()
        changed = deepcopy(payload)
        changed["config"]["runtime"]["bf16_policy"] = "bf16_storage"
        with patch.object(target.model, "load_state_dict") as load:
            with self.assertRaisesRegex(ValueError, "Resume configuration"):
                restore_checkpoint(changed, target, {}, generators={})
            load.assert_not_called()

    def test_replay_descriptor_is_frozen_and_policy_change_rejects_before_copy(self):
        with bf16_policy_scope("torch_amp", "cuda"):
            captured = operation_policy_spec()
        with bf16_policy_scope("bf16_storage", "cuda"):
            active = operation_policy_spec()
        self.assertEqual(captured[-1], BF16Policy())
        self.assertNotEqual(captured, active)
        self.assertEqual(_policy_description((False, None), captured, False)["bf16_policy"],
                         {"name": "torch_amp", "version": 1})
        # Metadata-only replay simulation, no CUDA graph execution claimed.
        graph = CapturedFunction(lambda x: x)
        graph._capture._state = "ready"
        graph._inputs = (torch.zeros(2),)
        graph._specs = (tensor_spec(graph._inputs[0]),)
        graph._autocast = (False, None)
        graph._operation_policy = captured
        graph._capture.replay = unittest.mock.Mock()
        with bf16_policy_scope("bf16_storage", "cuda"):
            with self.assertRaisesRegex(ValueError, "policy changed"):
                graph(torch.ones(2))
        graph._capture.replay.assert_not_called()

    def test_old_storage_subrecipe_cannot_exact_resume_before_loading(self):
        from deepopt.engine.checkpointing.checkpoint import _trainer_settings, _validate_resume_state

        # Host-only compatibility simulation. The Trainer CPU core is FP32;
        # actual BF16 training/resume belongs to the separate CUDA job.
        trainer = self.trainer()
        payload = deepcopy(_checkpoint_payload(trainer, trainer.config, generators={}, history=[]))
        trainer.model._deepopt_bf16_policy = BF16Policy("bf16_storage").as_dict()
        payload["config"]["runtime"]["bf16_policy"] = "bf16_storage"
        payload["trainer_settings"] = _trainer_settings(trainer)
        self.assertEqual(payload["trainer_settings"]["bf16_storage_recipe"]["primitive_recipe_version"], 3)
        payload["trainer_settings"]["bf16_storage_recipe"]["primitive_recipe_version"] = 2
        rng = torch.get_rng_state()
        with patch.object(trainer.model, "load_state_dict") as load:
            with self.assertRaisesRegex(ValueError, "numerical recipe"):
                _validate_resume_state(payload, trainer, {}, generators={}, train_loader=None,
                                       old_batch_size=None, requested_batch_size=None, saved_world=1)
            load.assert_not_called()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_replay_storage_signature_includes_primitive_and_attention_versions(self):
        from deepopt.cuda_graph.model import _storage_signature

        model = torch.nn.Linear(2, 1)
        model._deepopt_bf16_primitive_recipe = 3
        model._deepopt_attention_recipe_version = 2
        signature = _storage_signature(model)
        for marker in ("_deepopt_bf16_primitive_recipe", "_deepopt_attention_recipe_version"):
            setattr(model, marker, 1)
            self.assertNotEqual(_storage_signature(model), signature)
            setattr(model, marker, 3 if marker == "_deepopt_bf16_primitive_recipe" else 2)


if __name__ == "__main__":
    unittest.main()
