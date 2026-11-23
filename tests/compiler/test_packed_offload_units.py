"""SI-GB public aliases preserve the historical canonical byte descriptor."""

from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, localcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from deepopt._memory_units import normalize_packed_offload_units


class PackedOffloadUnitNormalizationTests(unittest.TestCase):
    def test_every_alias_uses_exact_SI_not_binary_GiB_and_owns_the_table(self):
        raw = dict(enabled=True, host_pinned_gb=80, device_staging_gb=4, d2h_inflight_gb=2,
                   host_pageable_gb="0.5", peak_budget_gb=Decimal("1.25"), copy_chunk_gb="0.016777216")
        before = deepcopy(raw)
        expected = dict(enabled=True, host_pinned_bytes=80_000_000_000, device_staging_bytes=4_000_000_000,
                        d2h_inflight_bytes=2_000_000_000, host_pageable_bytes=500_000_000,
                        peak_budget_bytes=1_250_000_000, copy_chunk_bytes=16_777_216)
        self.assertEqual(normalize_packed_offload_units(raw), expected)
        self.assertEqual(raw, before)
        self.assertNotEqual(expected["host_pinned_bytes"], 80 << 30)

    def test_default_80GB_is_only_for_enabled_GB_routes_missing_a_pinned_quantity(self):
        self.assertEqual(normalize_packed_offload_units(dict(enabled=True, device_staging_gb=4)),
                         dict(enabled=True, device_staging_bytes=4_000_000_000, host_pinned_bytes=80_000_000_000))
        for raw in ({}, {"enabled": True}, {"enabled": False}, {"enabled": False, "host_pinned_bytes": 123},
                    {"host_pinned_bytes": 1 << 100}, {"peak_budget_bytes": None}):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_packed_offload_units(raw), raw)
        disabled = normalize_packed_offload_units(dict(enabled=False, device_staging_gb=4))
        self.assertNotIn("host_pinned_bytes", disabled)
        explicit = normalize_packed_offload_units(dict(enabled=True, host_pinned_bytes=17, device_staging_gb=4))
        self.assertEqual(explicit["host_pinned_bytes"], 17)

    def test_conflicting_units_never_choose_even_equal_or_null_values(self):
        for stem in ("host_pinned", "device_staging", "d2h_inflight", "host_pageable", "peak_budget", "copy_chunk"):
            for byte_value in (1_000_000_000, None):
                with self.subTest(stem=stem, byte_value=byte_value), self.assertRaisesRegex(ValueError, "both"):
                    normalize_packed_offload_units({stem+"_gb": 1, stem+"_bytes": byte_value})

    def test_fractional_bytes_nonfinite_boolean_and_invalid_metadata_are_rejected(self):
        for value in (True, False, None, -1, "-0.000000001", "nan", float("inf"), "-Infinity",
                      "0.0000000001", Decimal("1.0000000001"), "not a number", object()):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                normalize_packed_offload_units({"host_pinned_gb": value})
        for value in ([], None, {1: 2}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_packed_offload_units(value)
        self.assertEqual(normalize_packed_offload_units({"host_pinned_gb": "0.000000001"}), {"host_pinned_bytes": 1})
        self.assertEqual(normalize_packed_offload_units({"host_pinned_gb": "-0.0"}), {"host_pinned_bytes": 0})

    def test_signed64_alias_limit_and_extreme_exponents_are_checked_without_rounding(self):
        with localcontext() as current:
            current.prec = 3
            self.assertEqual(normalize_packed_offload_units({"host_pinned_gb": "9223372036.854775807"}),
                             {"host_pinned_bytes": (1 << 63)-1})
            for value in ("9223372036.854775808", "1e1000000000", "1e-1000000000"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    normalize_packed_offload_units({"host_pinned_gb": value})

    def test_helper_and_public_resolver_are_Torch_free_in_a_real_subprocess(self):
        package_root = Path(sys.modules["deepopt"].__file__).resolve().parent.parent
        environment = {**os.environ, "PYTHONPATH": str(package_root),
                       "PYTHONDONTWRITEBYTECODE": "1"}
        completed = subprocess.run([sys.executable, "-S", "-c", """
import sys
from deepopt._memory_units import normalize_packed_offload_units
from deepopt._packed_offload import resolve_packed_offload
raw = {'version':2,'enabled':True,'device_staging_gb':4,'d2h_inflight_gb':2}
assert normalize_packed_offload_units(raw)['host_pinned_bytes'] == 80_000_000_000
assert resolve_packed_offload(raw).host_pinned_bytes == 80_000_000_000
assert 'torch' not in sys.modules and 'triton' not in sys.modules
"""], env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class PackedOffloadUnitConfigTests(unittest.TestCase):
    def config(self, policy):
        return dict(name="GB-config", seed=0, training=dict(epochs=1, batch_size=192, learning_rate=.01),
                    runtime=dict(device="cuda:0", precision="bf16", packed_activation_offload=deepcopy(policy),
                        compile=dict(enabled=True, backend="deepopt", fullgraph=True,
                                     options=dict(packed_activation_offload=deepcopy(policy)))))

    def test_transport_limits_default_to_auto_and_can_reset_inherited_caps(self):
        from deepopt._config_resolution import resolve_config
        from deepopt.config import config_from_dict

        raw = dict(version=2, enabled=True, host_pinned_gb=80)
        config = config_from_dict(self.config(raw))
        policy = config.runtime.packed_activation_offload
        self.assertEqual((policy.device_staging_bytes, policy.d2h_inflight_bytes), (0, 0))
        self.assertEqual(config_from_dict(json.loads(json.dumps(asdict(config)))), config)

        base = self.config(dict(raw, device_staging_gb=4, d2h_inflight_gb=3))
        incoming = self.config(dict(version=2, enabled=True,
                                    device_staging_gb="auto", d2h_inflight_bytes="auto"))
        merged = resolve_config(incoming, base=base)["runtime"]
        self.assertEqual(merged["packed_activation_offload"], policy.as_config())
        self.assertEqual(merged["compile"]["options"]["packed_activation_offload"], policy.as_config())

    def test_public_config_and_descriptor_roundtrip_canonicalize_to_unchanged_bytes(self):
        from deepopt._packed_offload import resolve_packed_offload
        from deepopt.config import config_from_dict

        raw = dict(version=2, enabled=True, host_pinned_gb=80, device_staging_gb=4, d2h_inflight_gb=2)
        parsed = config_from_dict(self.config(raw))
        self.assertEqual(parsed.runtime.packed_activation_offload.host_pinned_bytes, 80_000_000_000)
        self.assertEqual(parsed.runtime.packed_activation_offload.device_staging_bytes, 4_000_000_000)
        self.assertEqual(parsed.runtime.packed_activation_offload.d2h_inflight_bytes, 2_000_000_000)
        self.assertEqual(parsed.runtime.compile.options["packed_activation_offload"],
                         parsed.runtime.packed_activation_offload.as_config())
        canonical = json.loads(json.dumps(asdict(parsed)))
        self.assertEqual(config_from_dict(canonical), parsed)
        self.assertFalse(any(key.endswith("_gb") for key in parsed.runtime.packed_activation_offload.as_config()))
        self.assertEqual(resolve_packed_offload(parsed.runtime.packed_activation_offload.as_config()),
                         parsed.runtime.packed_activation_offload)

    def test_partial_GB_override_normalizes_before_inherited_byte_defaults_merge(self):
        from deepopt._config_resolution import resolve_config
        from deepopt._packed_offload import resolve_packed_offload

        canonical = resolve_packed_offload(dict(version=2, enabled=True, host_pinned_bytes=17,
            device_staging_bytes=19, d2h_inflight_bytes=23)).as_config()
        base = self.config(canonical)
        incoming = {"runtime": {"packed_activation_offload": {"host_pinned_gb": 80},
                    "compile": {"options": {"packed_activation_offload": {"host_pinned_gb": 80}}}}}
        before_base, before_incoming = deepcopy(base), deepcopy(incoming)
        resolved = resolve_config(incoming, base=base)
        policy = resolved["runtime"]["packed_activation_offload"]
        self.assertEqual(policy["host_pinned_bytes"], 80_000_000_000)
        self.assertEqual(policy["device_staging_bytes"], 19)
        self.assertEqual(policy["d2h_inflight_bytes"], 23)
        self.assertEqual(resolved["runtime"]["compile"]["options"]["packed_activation_offload"], policy)
        self.assertEqual(base, before_base)
        self.assertEqual(incoming, before_incoming)

    def test_conflicts_and_unknown_aliases_stay_errors_and_byte_only_defaults_stay_legacy(self):
        from deepopt._config_resolution import resolve_config
        from deepopt._packed_offload import resolve_packed_offload
        from deepopt.config import ConfigError, config_from_dict

        default = resolve_packed_offload()
        self.assertFalse(default.enabled)
        self.assertEqual(default.host_pinned_bytes, 0)
        self.assertNotIn("version", default.as_config())
        for raw in (dict(host_pinned_gb=1, host_pinned_bytes=1_000_000_000), dict(host_pinned_gib=80)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                resolve_packed_offload(raw)
            with self.subTest(config=raw), self.assertRaises(ConfigError):
                config_from_dict(self.config(raw))
        base = self.config(default.as_config())
        with self.assertRaises(ValueError):
            resolve_config({"runtime": {"packed_activation_offload": {"host_pinned_gb": 1, "host_pinned_bytes": 1}}}, base=base)

    def test_partial_enabled_GB_route_preserves_positive_inherited_host_and_fills_default_zero(self):
        from deepopt._config_resolution import resolve_config
        from deepopt._packed_offload import resolve_packed_offload

        for inherited, expected in ((17, 17), (0, 80_000_000_000)):
            with self.subTest(inherited_host=inherited):
                canonical = resolve_packed_offload(dict(version=2, enabled=bool(inherited),
                    host_pinned_bytes=inherited, device_staging_bytes=19, d2h_inflight_bytes=23)).as_config()
                base = self.config(canonical)
                partial = {"enabled": True, "device_staging_gb": 4}
                incoming = {"runtime": {"packed_activation_offload": deepcopy(partial),
                    "compile": {"options": {"packed_activation_offload": deepcopy(partial)}}}}
                before_base, before_incoming = deepcopy(base), deepcopy(incoming)
                result = resolve_config(incoming, base=base)
                selected = result["runtime"]["packed_activation_offload"]
                self.assertEqual(selected["host_pinned_bytes"], expected)
                self.assertEqual(selected["device_staging_bytes"], 4_000_000_000)
                self.assertEqual(selected["d2h_inflight_bytes"], 23)
                self.assertEqual(result["runtime"]["compile"]["options"]["packed_activation_offload"], selected)
                self.assertEqual(base, before_base)
                self.assertEqual(incoming, before_incoming)


if __name__ == "__main__":
    unittest.main()
