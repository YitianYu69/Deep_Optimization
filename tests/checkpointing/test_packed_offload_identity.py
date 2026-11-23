"""Transport metadata is serializable; historical absent policy is disabled."""

import json
import unittest

from deepopt._packed_offload import PackedActivationOffloadPolicy
from deepopt.engine.checkpointing.identity import compatibility_config


class PackedOffloadCheckpointIdentityTests(unittest.TestCase):
    def test_historical_absent_and_explicit_disabled_transport_are_compatible(self):
        historical = {"runtime": {"device": "cpu", "precision": "fp32"}}
        current = {"runtime": {**historical["runtime"],
                              "packed_activation_offload": PackedActivationOffloadPolicy().as_config()}}
        self.assertEqual(compatibility_config(historical), compatibility_config(current))

    def test_enabled_policy_is_explicit_metadata_and_preserves_strict_resume_identity(self):
        policy = PackedActivationOffloadPolicy(enabled=True, host_pinned_bytes=1 << 20,
            device_staging_bytes=1 << 20, d2h_inflight_bytes=1 << 20)
        enabled = compatibility_config({"runtime": {"packed_activation_offload": policy.as_config()}})
        disabled = compatibility_config({"runtime": {}})
        self.assertNotEqual(enabled, disabled)
        serialized = json.loads(json.dumps(enabled, allow_nan=False))
        self.assertEqual(serialized["runtime"]["packed_activation_offload"], policy.as_dict())
        self.assertTrue(all(type(value) in (str, int, float, bool, type(None))
                            for value in serialized["runtime"]["packed_activation_offload"].values()))


if __name__ == "__main__":
    unittest.main()
