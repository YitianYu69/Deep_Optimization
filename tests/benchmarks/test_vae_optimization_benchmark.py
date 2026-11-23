"""Pinned compiler experiments keep wheel migration separate from pass effects."""

from contextlib import redirect_stdout
from copy import deepcopy
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from examples.vae import benchmark_optimization as experiment, native_execution
from tests.benchmarks.test_vae_execution_benchmark import _successful_worker, _worker_args
from tests.benchmarks.test_vae_native_execution import _receipt
from tests.benchmarks.test_vae_memory_comparison import audited_row


def _manifest(directory, *, convolution=False, fusion=False,
              norm_stage="forward_backward", swiglu_stage="forward_backward"):
    control = {"version": experiment.CONTROL_VERSION, "source_sha256": experiment.CONTROL_SOURCE_SHA,
               "files_hashed": 457}
    candidate = {"version": "0.1.0+nativeautodiff.20261120.vaeopt.test", "source_sha256": "1" * 64,
                 "files_hashed": 460}
    arms = {}
    for name in "ABC":
        arms[name] = {"wheel": {"path": "control.whl" if name == "A" else "candidate.whl",
            "sha256": experiment.CONTROL_WHEEL_SHA if name == "A" else "2" * 64},
            "install_root": "control" if name == "A" else "candidate",
            "expected_installed": deepcopy(control if name == "A" else candidate),
            "native_options": {} if name == "A" else {key: name == "C" for key in experiment.PASS_OPTIONS}}
        if convolution and name != "A":
            arms[name]["native_options"] = dict(experiment.CONVOLUTION_DEFAULTS)
            if name == "C":
                arms[name]["native_options"].update(native_conv_backward_order="dw_first", cuda_conv_dx_geometry=True)
        if fusion and name != "A":
            arms[name]["native_options"] = dict(experiment.OPTION_DEFAULTS)
            if name == "C":
                arms[name]["native_options"].update(fuse_norm_activation_residuals=norm_stage != "off",
                    norm_activation_fusion_stage=norm_stage, fuse_swiglu_residuals=swiglu_stage != "off",
                    swiglu_fusion_stage=swiglu_stage)
    path = Path(directory) / "experiment.json"
    value = {"schema_version": 1, "arms": arms}
    if fusion:
        value["experiment_profile"] = "native_training_fusion_v3"
    path.write_text(json.dumps(value))
    return path, value


def _workers(directory, blocks=1, *, convolution=False, fusion=False,
             norm_stage="forward_backward", swiglu_stage="forward_backward"):
    receipt, _ = _receipt(directory)
    path, manifest = _manifest(directory, convolution=convolution, fusion=fusion,
        norm_stage=norm_stage, swiglu_stage=swiglu_stage)
    identities, workers = {}, []
    for block in range(1, blocks + 1):
        for name in "ABC":
            spec = manifest["arms"][name]
            pin = Path(directory) / f"pin-{block}-{name}.json"
            options = Path(directory) / f"options-{block}-{name}.json"
            options.write_text(json.dumps(spec["native_options"]))
            pin.write_text(json.dumps({"schema_version": 1, "arm": name,
                "expected_installed": spec["expected_installed"],
                "native_options_sha256": hashlib.sha256(options.read_bytes()).hexdigest() if name != "A" else None}))
            args = _worker_args(receipt, Path(directory) / f"{block}-{name}.json", "native")
            args.arm_manifest = pin
            args.native_options_json = options if name != "A" else None
            plan = native_execution.execution_plan(args)
            identity = {"distribution_name": "deep-optimization", **spec["expected_installed"],
                "package_path": str(Path(directory) / spec["install_root"] / "deepopt"),
                "import_path": str(Path(directory) / spec["install_root"] / "deepopt/__init__.py"),
                "install_root": str(Path(directory) / spec["install_root"]), "editable": False}
            identities[name] = identity
            row = _successful_worker(plan, "native", identity=identity)
            row['memory_audit'] = audited_row(arm=name, pid=1000 + block * 3 + ord(name))['memory_audit']
            row.update(arm=name, block=block, worker_exit_code=0, expected_settings=deepcopy(plan["settings"]))
            row['compiler_report_summary']['effective_compiler_options'] = [deepcopy(spec['native_options'])]
            row['compiler_report_summary']['fusion_candidates'] = [
                {'pattern': pattern, 'selected': True} for option, pattern in experiment.PASS_PATTERNS.items()
                if experiment.option_enabled(option, spec['native_options'].get(option, experiment.OPTION_DEFAULTS[option]))]
            for option, stage_option in experiment.FUSION_STAGE_OPTIONS.items():
                if spec['native_options'].get(option):
                    row['compiler_report_summary']['fusion_candidates'].extend(
                        {'pattern': pattern, 'execution_phase': experiment.FUSION_PATTERN_PHASES[pattern], 'selected': True}
                        for pattern in experiment.FUSION_STAGE_PATTERNS[option][spec['native_options'][stage_option]])
            row["provenance"].update(deepopt_source_sha256=identity["source_sha256"],
                source_sha256=identity["source_sha256"], deepopt_version=identity["version"],
                deepopt_path=identity["package_path"])
            workers.append(row)
    return path, manifest, identities, workers


class OptimizationManifestTests(unittest.TestCase):
    def test_fusion_profile_requires_exact_fields_marker_defaults_and_flag_stage_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path, original = _manifest(directory, fusion=True)
            self.assertEqual(experiment.load_manifest(path)[0], original)
            self.assertEqual(original["arms"]["B"]["native_options"], experiment.OPTION_DEFAULTS)
            for failure in ("missing_marker", "wrong_marker", "partial_fields", "conv_enabled", "legacy_enabled",
                    "enabled_off", "disabled_forward", "norm_activation_pack", "boolean_stage", "integer_flag", "B_stage"):
                changed = deepcopy(original)
                after = changed["arms"]["C"]["native_options"]
                if failure == "missing_marker":
                    changed.pop("experiment_profile")
                elif failure == "wrong_marker":
                    changed["experiment_profile"] = "native_convolution_wave2"
                elif failure == "partial_fields":
                    after.pop("swiglu_fusion_stage")
                elif failure == "conv_enabled":
                    after["native_conv_backward_order"] = "dw_first"
                elif failure == "legacy_enabled":
                    after["fuse_norm_cat_body"] = True
                elif failure == "enabled_off":
                    after["norm_activation_fusion_stage"] = "off"
                elif failure == "disabled_forward":
                    after["fuse_swiglu_residuals"] = False
                elif failure == "norm_activation_pack":
                    after["norm_activation_fusion_stage"] = "activation_pack"
                elif failure == "boolean_stage":
                    after["swiglu_fusion_stage"] = True
                elif failure == "integer_flag":
                    after["fuse_swiglu_residuals"] = 1
                else:
                    changed["arms"]["B"]["native_options"]["swiglu_fusion_stage"] = "forward"
                path.write_text(json.dumps(changed))
                with self.subTest(failure=failure), self.assertRaises(ValueError):
                    experiment.load_manifest(path)

    def test_fusion_ablation_stages_remain_explicit_and_keep_all_legacy_options_off(self):
        for norm, swiglu in (("forward", "off"), ("off", "activation_pack"), ("off", "forward"),
                             ("forward_backward", "off"), ("off", "forward_backward"), ("forward", "forward")):
            with self.subTest(norm=norm, swiglu=swiglu), tempfile.TemporaryDirectory() as directory:
                path, value = _manifest(directory, fusion=True, norm_stage=norm, swiglu_stage=swiglu)
                self.assertEqual(experiment.load_manifest(path)[0], value)
                self.assertTrue(all(value["arms"]["C"]["native_options"][key] == default
                    for key, default in experiment.CONVOLUTION_DEFAULTS.items()))

    def test_convolution_profile_preserves_strict_enum_and_legacy_pass_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path, value = _manifest(directory, convolution=True)
            self.assertEqual(experiment.load_manifest(path)[0], value)
            for option, replacement in (("native_conv_backward_order", True),
                    ("native_conv_backward_order", "dx_first"), ("cuda_conv_dx_geometry", 1),
                    ("fuse_norm_cat_body", True), ("donate_fresh_backward_add", True)):
                changed = deepcopy(value)
                changed["arms"]["C"]["native_options"][option] = replacement
                path.write_text(json.dumps(changed))
                with self.subTest(option=option, value=replacement), self.assertRaises(ValueError):
                    experiment.load_manifest(path)
            for selected in ({"native_conv_backward_order": "dw_first"}, {"cuda_conv_dx_geometry": True}):
                changed = deepcopy(value)
                changed["arms"]["C"]["native_options"] = dict(experiment.CONVOLUTION_DEFAULTS, **selected)
                path.write_text(json.dumps(changed))
                self.assertEqual(experiment.load_manifest(path)[0], changed)

    def test_pins_immutable_control_and_same_candidate_with_explicit_off_on_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            path, value = _manifest(directory)
            self.assertEqual(experiment.load_manifest(path)[0], value)
            cases = ((('A', 'wheel', 'sha256'), 'a' * 64),
                     (('A', 'expected_installed', 'files_hashed'), 458),
                     (('B', 'native_options', 'fuse_norm_cat_body'), True),
                     (('C', 'native_options', 'other_pass'), True),
                     (('C', 'expected_installed', 'version'), 'other'),
                     (('C', 'expected_installed', 'files_hashed'), True))
            for keys, replacement in cases:
                with self.subTest(keys=keys):
                    changed = deepcopy(value)
                    target = changed['arms']
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = replacement
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        experiment.load_manifest(path)
            path.write_text('{"schema_version":1,"schema_version":1,"arms":{}}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                experiment.load_manifest(path)

    def test_wheel_and_installed_byte_membership_are_verified_without_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'installed'
            package = target / 'deepopt'
            package.mkdir(parents=True)
            files = {'__init__.py': b'raise RuntimeError("must not import")\n',
                     'nested/value.txt': b'payload\n'}
            wheel = root / 'candidate.whl'
            metadata = Message()
            metadata['Name'], metadata['Version'] = 'deep-optimization', '0.1.0+test'
            with zipfile.ZipFile(wheel, 'w') as archive:
                for name, data in files.items():
                    path = package / name
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(data)
                    archive.writestr('deepopt/' + name, data)
                archive.writestr('deep_optimization-0.1.0+test.dist-info/METADATA', str(metadata))
            digest = hashlib.sha256()
            for path in sorted(package.rglob('*')):
                if path.is_file():
                    digest.update(path.relative_to(package).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
            arm = {'wheel': {'path': str(wheel), 'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()},
                'install_root': str(target), 'expected_installed': {'version': '0.1.0+test',
                    'source_sha256': digest.hexdigest(), 'files_hashed': 2}}
            self.assertFalse(experiment.verify_arm(arm, root / 'experiment.json')['editable'])
            (package / 'extra.py').write_text('extra = True')
            with self.assertRaisesRegex(ValueError, 'differs from its wheel'):
                experiment.verify_arm(arm, root / 'experiment.json')


class OptimizationAggregationTests(unittest.TestCase):
    def test_fusion_stage_requires_both_exact_executors_and_effective_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identities, rows = _workers(directory, blocks=3, fusion=True)
            summary = experiment.aggregate_blocks(rows, identities, 3)
            self.assertTrue(summary["valid"], summary["invalid_reasons"])
            self.assertTrue(summary["target_CUDA_evidence_complete"])
            passes = summary["pass_selections"][2]["passes"]
            norm = passes["fuse_norm_activation_residuals"]
            self.assertEqual(norm["requested_stage"], "forward_backward")
            self.assertEqual(norm["selected_by_pattern"], {"group_norm_silu_residual_forward": 1,
                "group_norm_silu_residual_backward": 1})
            self.assertTrue(norm["selection_complete"])
            self.assertTrue(passes["fuse_swiglu_residuals"]["selection_complete"])
            for failure in ("norm_backward_missing", "swiglu_backward_missing", "wrong_stage_executor",
                    "wrong_phase", "effective_stage", "missing_effective_default", "selected_while_off"):
                changed = deepcopy(rows)
                compiler = changed[-1]["compiler_report_summary"]
                candidates = compiler["fusion_candidates"]
                if failure in ("norm_backward_missing", "swiglu_backward_missing"):
                    pattern = "group_norm_silu_residual_backward" if failure.startswith("norm") else "swiglu_backward_residuals"
                    compiler["fusion_candidates"] = [row for row in candidates if row["pattern"] != pattern]
                elif failure == "wrong_stage_executor":
                    compiler["fusion_candidates"].append({"selected": True, "pattern": "swiglu_activation_pack",
                                                         "execution_phase": "forward"})
                elif failure == "wrong_phase":
                    next(row for row in candidates if row["pattern"] == "swiglu_backward_residuals")["execution_phase"] = "forward"
                elif failure == "effective_stage":
                    compiler["effective_compiler_options"][0]["norm_activation_fusion_stage"] = "forward"
                elif failure == "missing_effective_default":
                    changed[-2]["compiler_report_summary"]["effective_compiler_options"][0].pop("swiglu_fusion_stage")
                else:
                    changed[-2]["compiler_report_summary"]["fusion_candidates"] = deepcopy(candidates)
                with self.subTest(failure=failure):
                    invalid = experiment.aggregate_blocks(changed, identities, 3)
                    self.assertFalse(invalid["valid"])
                    self.assertFalse(invalid["ratios"])
                    self.assertFalse(invalid["target_CUDA_evidence_complete"])
                    self.assertTrue(invalid["invalid_reasons"])

    def test_forward_and_activation_pack_profiles_reject_other_stage_selections(self):
        for norm, swiglu, forbidden in (("forward", "forward", "group_norm_silu_residual_backward"),
                ("off", "activation_pack", "swiglu_forward_residuals")):
            with self.subTest(norm=norm, swiglu=swiglu), tempfile.TemporaryDirectory() as directory:
                _, _, identities, rows = _workers(directory, fusion=True, norm_stage=norm, swiglu_stage=swiglu)
                self.assertTrue(experiment.aggregate_blocks(rows, identities, 1)["valid"])
                rows[-1]["compiler_report_summary"]["fusion_candidates"].append({"selected": True,
                    "pattern": forbidden, "execution_phase": experiment.FUSION_PATTERN_PHASES[forbidden]})
                invalid = experiment.aggregate_blocks(rows, identities, 1)
                self.assertFalse(invalid["valid"])
                self.assertFalse(invalid["ratios"])

    def test_typed_convolution_selection_and_memory_isolation_qualify_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identities, rows = _workers(directory, convolution=True)
            summary = experiment.aggregate_blocks(rows, identities, 1)
            self.assertTrue(summary["valid"], summary["invalid_reasons"])
            passes = summary["pass_selections"][-1]["passes"]
            self.assertEqual(passes["native_conv_backward_order"]["requested_value"], "dw_first")
            self.assertTrue(passes["native_conv_backward_order"]["requested"])
            self.assertFalse(passes["donate_fresh_backward_add"]["requested"])
            for failure in ("free_memory_drift", "cleanup_failed", "bool_effective_enum"):
                changed = deepcopy(rows)
                if failure == "free_memory_drift":
                    changed[-1]["memory_audit"]["snapshots"]["before_model"]["device_free_bytes"] -= 2**30
                elif failure == "cleanup_failed":
                    changed[-1]["memory_audit"]["cleanup"]["completed"] = False
                else:
                    changed[-1]["compiler_report_summary"]["effective_compiler_options"][0]["native_conv_backward_order"] = True
                with self.subTest(failure=failure):
                    invalid = experiment.aggregate_blocks(changed, identities, 1)
                    self.assertFalse(invalid["valid"])
                    self.assertFalse(invalid["ratios"])
                    self.assertTrue(invalid["invalid_reasons"])

    def test_three_blocks_are_three_samples_and_wheel_migration_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, identities, rows = _workers(directory, blocks=3)
            for row in rows:
                row['measurement']['wall_ms_per_update'] = {'A': 12., 'B': 10., 'C': 8.}[row['arm']]
            summary = experiment.aggregate_blocks(rows, identities, 3)
        self.assertTrue(summary['valid'], summary['invalid_reasons'])
        self.assertAlmostEqual(summary['ratios']['wheel_migration_A_B']['runtime_speedup_left_over_right']['median'], 1.2)
        ratio = summary['ratios']['optimization_B_C']['runtime_speedup_left_over_right']
        self.assertEqual(ratio['median'], 1.25)
        self.assertEqual(ratio['samples'], 3)
        self.assertFalse(summary['performance_policy_promoted'])

    def test_whole_cohort_invalidates_ratio_for_failures_recipe_drift_and_cross_block_gpu_drift(self):
        for failure in ('nonzero_exit', 'different_optimizer', 'unplanned_flag', 'gpu_drift', 'missing_arm',
                        'unselected_pass', 'wrong_effective_pass', 'pass_selected_while_disabled'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                _, _, identities, rows = _workers(directory, blocks=2)
                if failure == 'nonzero_exit':
                    rows[-1]['worker_exit_code'] = 1
                elif failure == 'different_optimizer':
                    rows[-1]['provenance']['optimizer_source_sha256'] = 'f' * 64
                elif failure == 'unplanned_flag':
                    rows[-1]['settings']['runtime']['compile']['options']['debug'] = True
                elif failure == 'gpu_drift':
                    for row in rows[3:]:
                        row['software']['gpu']['uuid'] = 'another-GPU'
                elif failure == 'unselected_pass':
                    rows[-1]['compiler_report_summary']['fusion_candidates'] = []
                elif failure == 'wrong_effective_pass':
                    rows[-1]['compiler_report_summary']['effective_compiler_options'][0]['fuse_norm_cat_body'] = False
                elif failure == 'pass_selected_while_disabled':
                    rows[-2]['compiler_report_summary']['fusion_candidates'] = deepcopy(
                        rows[-1]['compiler_report_summary']['fusion_candidates'])
                else:
                    rows.pop()
                summary = experiment.aggregate_blocks(rows, identities, 2)
                self.assertFalse(summary['valid'])
                self.assertFalse(summary['ratios'])
                self.assertTrue(summary['invalid_reasons'])
                self.assertFalse(summary['target_CUDA_evidence_complete'])

    def test_supervisor_preserves_peer_receipts_and_checks_its_sources(self):
        for failure in ('worker_exit', 'supervisor_source', 'malformed_worker', 'aggregation_failure'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                manifest_path, manifest, identities, rows = _workers(directory)
                args = experiment.parser().parse_args(['--experiment-manifest', str(manifest_path),
                    '--from-result', rows[0]['origin']['path'], '--output', str(Path(directory) / 'result.json'),
                    '--warmup', '1', '--updates', '2', '--trials', '1'])
                # These are simulated workers for failure/receipt rules, never target execution evidence.
                source = str(Path(experiment.__file__).relative_to(experiment.ROOT))
                plan = {'arms': manifest['arms'], 'installed_identities': identities, 'order': [list('ABC')],
                    'origin': rows[0]['origin'], 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    'supervisor_sources': {source: hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest()}}

                def worker(command, **kwargs):
                    worker_args = native_execution.parser().parse_args(command[3:])
                    worker_plan = native_execution.execution_plan(worker_args)
                    name = worker_plan['experiment_arm']['manifest']['arm']
                    row = deepcopy(next(row for row in rows if row['arm'] == name))
                    row['settings'] = worker_plan['settings']
                    if name == 'A' and failure == 'malformed_worker':
                        row['compiler_report_summary'] = None
                    worker_args.output.write_text(json.dumps(row))
                    if name == 'C' and failure == 'supervisor_source':
                        plan['supervisor_sources'][source] = '0' * 64
                    return SimpleNamespace(returncode=1 if failure == 'worker_exit' and name == 'A' else 0)

                original_aggregate = experiment.aggregate_blocks
                def aggregate(*arguments, **keywords):
                    if failure == 'aggregation_failure':
                        raise ValueError('injected aggregate failure')
                    return original_aggregate(*arguments, **keywords)

                with patch.object(experiment, 'experiment_plan', return_value=plan), \
                     patch.object(experiment, 'aggregate_blocks', side_effect=aggregate), \
                     patch.object(experiment, 'verify_arm', side_effect=lambda arm, path:
                         identities['A' if arm['expected_installed']['version'] == experiment.CONTROL_VERSION else 'B']), \
                     patch.object(experiment.subprocess, 'run', side_effect=worker), redirect_stdout(io.StringIO()):
                    code = experiment.run_experiment(args)
                report = json.loads(args.output.read_text())
                self.assertEqual(code, 1)
                self.assertEqual(len(report['results']), 3)
                self.assertFalse(report['summary']['valid'])
                self.assertFalse(report['summary']['ratios'])
                self.assertTrue((args.output.parent / report['experiment_manifest_artifact']['path']).is_file())


if __name__ == '__main__':
    unittest.main()
