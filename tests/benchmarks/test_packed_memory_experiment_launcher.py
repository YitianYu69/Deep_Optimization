"""Strict same-candidate manifests and installed-only launch boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.compiler import run_packed_memory_experiments as launcher


class PackedMemoryExperimentLauncherTests(unittest.TestCase):
    def test_entry_point_help_and_manifest_builder_are_torch_free(self):
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-S", "-c", "import sys; from benchmarks.compiler import run_packed_memory_experiments as m; m.parser(); assert 'torch' not in sys.modules"]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_profiles_normalize_without_weakening_historical_option_contract(self):
        from examples.vae.benchmark_optimization import load_manifest, OPTION_DEFAULTS

        candidate = {"wheel": {"path": "/explicit/candidate.whl", "sha256": "a" * 64},
            "install_root": "/explicit/candidate-install",
            "expected_installed": {"version": "1+candidate", "source_sha256": "b" * 64, "files_hashed": 8}}
        before = deepcopy(candidate)
        for profile in launcher.PROFILES:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                args = SimpleNamespace(experiment=profile, dx_batch_size=16, selection_mode="peak", no_tiling=False,
                    host_pinned_mib=65536, device_staging_mib=4096, d2h_inflight_mib=2048)
                manifest = launcher.experiment(candidate, args)
                path = Path(temporary) / "experiment.json"
                path.write_text(json.dumps(manifest))
                normalized, digest = load_manifest(path)
                self.assertEqual(set(normalized["arms"]), set(launcher.PROFILES[profile][1]))
                self.assertEqual(len(digest), 64)
                self.assertTrue(all(arm["native_options"]["fuse_norm_activation_residuals"] is False
                                    for arm in manifest["arms"].values()))
                self.assertTrue(all(set(arm["native_options"]) == set(launcher.OPTIONS)
                                    for arm in manifest["arms"].values()))
        self.assertEqual(candidate, before)
        self.assertEqual(len(OPTION_DEFAULTS), 8)

    def test_physical_allocator_and_original_shape_contracts_have_fresh_process_groups(self):
        groups = launcher.contract_groups()
        names = [label for label, _, _ in groups]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue({"arena", "arena-original", "convolution", "convolution-original", "model-scope"}.issubset(names))
        original = {label: environment for label, _, environment in groups}
        self.assertEqual(original["arena-original"]["DEEPOPT_TEST_LARGE_CUDA_PINNED_ARENA"], "1")
        self.assertEqual(original["convolution-original"]["DEEPOPT_TEST_LARGE_CUDA_CONV_REUSE"], "1")
        self.assertEqual(groups[0][0], "convolution-original")
        self.assertEqual(len(launcher.contract_groups(include_convolution=False)), 4)
        self.assertFalse(any(label.startswith("convolution") for label, _, _ in
                             launcher.contract_groups(include_convolution=False)))


if __name__ == "__main__":
    unittest.main()
