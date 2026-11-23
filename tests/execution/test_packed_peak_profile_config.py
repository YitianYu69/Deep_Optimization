"""Public config handoff to the immutable compiler peak profile."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from deepopt.config import ConfigError, compile_config_from_dict, config_from_dict


def profile():
    return {"schema_version": 1, "scope_identity": "a"*64, "cuda_execution_verified": True,
        "source_receipt_sha256": "b"*64, "windows": [
            {"region": 1, "phase": "backward", "boundary": 245, "baseline_bytes": 105242864128}]}


def config(value):
    policy = dict(version=2, enabled=True, host_pinned_bytes=4096,
                  device_staging_bytes=4096, d2h_inflight_bytes=4096)
    return {"name": "target-peak-profile", "seed": 0,
        "training": {"epochs": 1, "batch_size": 1, "learning_rate": .01},
        "runtime": {"device": "cuda:0", "precision": "bf16", "packed_activation_offload": policy,
            "compile": {"enabled": True, "backend": "deepopt", "fullgraph": True,
                "options": {"packed_activation_offload": deepcopy(policy),
                            "packed_offload_peak_profile": value}}}}


class PackedPeakProfileConfigTests(unittest.TestCase):
    def test_Torch_free_schema_accepts_only_the_explicit_structured_option(self):
        environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]/"src"),
                       "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-c", """
import sys
from deepopt.config import compile_config_from_dict
raw = {'schema_version':1,'scope_identity':'a'*64,'cuda_execution_verified':True,
    'source_receipt_sha256':'b'*64,'windows':[{'region':0,'phase':'forward','boundary':0,'baseline_bytes':0}]}
compiled = compile_config_from_dict({'backend':'deepopt','options':{'packed_offload_peak_profile':raw}})
assert compiled.options['packed_offload_peak_profile']['windows'][0]['source'] == 'target_storage_ledger'
assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
"""], env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_runtime_schema_JSON_roundtrip_and_backend_frozen_context_handoff(self):
        from deepopt._config_resolution import resolve_config
        from deepopt.compiler import BackendConfig
        from deepopt.compiler.registry import KernelCodegenContext

        raw = profile()
        parsed = config_from_dict(config(raw))
        raw["windows"][0]["baseline_bytes"] = 0
        self.assertEqual(parsed.runtime.compile.options["packed_offload_peak_profile"]["windows"][0]["baseline_bytes"],
                         105242864128)
        restored = config_from_dict(json.loads(json.dumps(asdict(parsed))))
        self.assertEqual(restored, parsed)
        owned = resolve_config(restored, base=None)
        backend = BackendConfig.from_options(owned["runtime"]["compile"]["options"])
        context = KernelCodegenContext(packed_offload_peak_profile=backend.packed_offload_peak_profile)
        self.assertEqual(context.packed_offload_peak_profile.windows[0].baseline_bytes, 105242864128)
        owned["runtime"]["compile"]["options"]["packed_offload_peak_profile"]["windows"][0]["baseline_bytes"] = 1
        self.assertEqual(context.packed_offload_peak_profile.windows[0].baseline_bytes, 105242864128)
        with self.assertRaises(FrozenInstanceError):
            context.packed_offload_peak_profile.windows[0].baseline_bytes = 2
        self.assertEqual(backend.packed_activation_offload.version, 2)

    def test_strict_top_level_and_window_fields_match_the_target_profile_contract(self):
        invalid = []
        for update in ({"schema_version": True}, {"schema_version": 2}, {"scope_identity": "X"*64},
                       {"source_receipt_sha256": "a"*63}, {"cuda_execution_verified": 1},
                       {"windows": []}, {"windows": "not rows"}, {"unknown": True}):
            invalid.append(profile() | update)
        for update in ({"region": True}, {"region": -1}, {"boundary": 1.5}, {"baseline_bytes": False},
                       {"phase": "gpu"}, {"source": "cold_logical_estimate"}, {"unknown": 0}):
            value = profile()
            value["windows"][0].update(update)
            invalid.append(value)
        value = profile()
        value["windows"][0].pop("boundary")
        invalid.append(value)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ConfigError):
                config_from_dict(config(value))

    def test_unknown_nested_options_and_other_compilers_remain_rejected(self):
        for options in ({"unknown_nested": {"x": 1}}, {"some_option": [1, 2]}):
            with self.subTest(options=options), self.assertRaises(ConfigError):
                compile_config_from_dict({"backend": "deepopt", "options": options})
        with self.assertRaises(ConfigError):
            compile_config_from_dict({"backend": "inductor", "options": {"packed_offload_peak_profile": profile()}})
        empty = compile_config_from_dict({"backend": "deepopt", "options": {"packed_offload_peak_profile": None}})
        self.assertIsNone(empty.options["packed_offload_peak_profile"])
        with self.assertRaisesRegex(ConfigError, "Choose runtime.compile.mode"):
            compile_config_from_dict({"backend": "deepopt", "mode": "default",
                                     "options": {"packed_offload_peak_profile": profile()}})


if __name__ == "__main__":
    unittest.main()
