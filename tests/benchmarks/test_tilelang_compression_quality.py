"""The CPU quality screen reports approximation rather than backend parity."""

import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[2] / "benchmarks/normalization/benchmark_tilelang_compression_quality.py"
SPEC = importlib.util.spec_from_file_location("tilelang_compression_quality", PATH)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


class CompressionQualityTests(unittest.TestCase):
    def test_quality_memory_tradeoff_and_moment_correction(self):
        report = quality.screen(rows=3, columns=1031, bits=(2, 8), groups=(1024,), seed=12)
        low, high = report["results"]
        self.assertEqual(report["execution"], "cpu_codec_quality")
        self.assertLess(high["xhat"]["relative_l2"], low["xhat"]["relative_l2"])
        self.assertGreater(high["saved_xhat_and_correction_bytes"], low["saved_xhat_and_correction_bytes"])
        for row in report["results"]:
            self.assertLess(row["mean_max_abs_error"], 1e-6)
            self.assertLess(row["second_moment_max_abs_error"], 1e-6)
            self.assertEqual(set(row["gradient_errors_vs_uncompressed"]), {"dx", "dweight", "dbias"})
            self.assertEqual(row["gradient_errors_vs_uncompressed"]["dbias"]["max_abs"], 0.)

    def test_reproducible_stress_distributions_and_realized_group(self):
        kwargs = dict(rows=2, columns=1031, bits=(4,), groups=(1024,),
                      distributions=("normal", "large_offset", "outliers"), seed=42)
        report = quality.screen(**kwargs)
        self.assertEqual(report, quality.screen(**kwargs))
        self.assertEqual(len(report["results"]), 3)
        self.assertTrue(all(row["realized_group_size"] == 1024 for row in report["results"]))

    def test_invalid_cases_are_rejected(self):
        for kwargs in (dict(rows=0), dict(columns=-1), dict(bits=(3,)), dict(groups=(0,))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                quality.screen(**kwargs)


if __name__ == "__main__":
    unittest.main()
