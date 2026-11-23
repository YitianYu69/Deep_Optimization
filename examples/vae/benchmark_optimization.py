"""Run pinned native compiler controls in fresh worker processes.

Historical A/B/C profiles isolate wheel migration from compiler passes.
Same-release D0/D1 isolates workspace reuse at fixed packed residency;
R0/BEST measures the declared combined demand point against resident execution.
"""

from datetime import datetime, timezone
from copy import deepcopy
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
CONTROL_VERSION = "0.1.0+nativeautodiff.20261119.vaememory.4"
CONTROL_WHEEL_SHA = "d2b7c417237211cbf8853c47d72028f1aa3cdf9d7782de715787dc345195ab94"
CONTROL_SOURCE_SHA = "fb5a619fad46e77831fd97ba00818af870db46feb6dc39bc3b32c6392cf09775"
PASS_OPTIONS = frozenset(("fuse_norm_cat_body", "donate_fresh_backward_add"))
CONVOLUTION_DEFAULTS = {"fuse_norm_cat_body": False, "donate_fresh_backward_add": False,
                        "native_conv_backward_order": "source", "cuda_conv_dx_geometry": False}
OPTION_DEFAULTS = {**CONVOLUTION_DEFAULTS, "fuse_norm_activation_residuals": False,
    "norm_activation_fusion_stage": "off", "fuse_swiglu_residuals": False, "swiglu_fusion_stage": "off"}
PASS_PATTERNS = {"fuse_norm_cat_body": "group_norm_cat_body_destination",
                 "donate_fresh_backward_add": "fresh_backward_add_donation",
                 "native_conv_backward_order": "native_conv_dw_first",
                 "cuda_conv_dx_geometry": "native_conv_cuda_geometry"}
FUSION_STAGE_OPTIONS = {"fuse_norm_activation_residuals": "norm_activation_fusion_stage",
                        "fuse_swiglu_residuals": "swiglu_fusion_stage"}
FUSION_STAGE_PATTERNS = {
    "fuse_norm_activation_residuals": {"off": (), "forward": ("group_norm_silu_residual_forward",),
        "forward_backward": ("group_norm_silu_residual_forward", "group_norm_silu_residual_backward")},
    "fuse_swiglu_residuals": {"off": (), "activation_pack": ("swiglu_activation_pack",),
        "forward": ("swiglu_forward_residuals",),
        "forward_backward": ("swiglu_forward_residuals", "swiglu_backward_residuals")}}
FUSION_PATTERN_PHASES = {"group_norm_silu_residual_forward": "forward",
    "group_norm_silu_residual_backward": "backward", "swiglu_activation_pack": "forward",
    "swiglu_forward_residuals": "forward", "swiglu_backward_residuals": "backward"}
OPTION_PROFILES = {"legacy_memory_wave1": {key: OPTION_DEFAULTS[key] for key in PASS_OPTIONS},
    "native_convolution_wave2": CONVOLUTION_DEFAULTS, "native_training_fusion_v3": OPTION_DEFAULTS}
WORKSPACE_PROFILE = "native_convolution_workspace_v2"
COMBINED_PROFILE = "native_memory_combined_v2"
SELECTION_PROFILE = "native_offload_selection_v2"
ARENA_PROFILE = "native_pinned_arena_v2"
MEMORY_ARMS = {WORKSPACE_PROFILE: ("D0", "D1"), COMBINED_PROFILE: ("R0", "BEST"),
    SELECTION_PROFILE: ("S0", "S1"), ARENA_PROFILE: ("H0", "H1")}
PAIRED_MEMORY_PROFILES = tuple(MEMORY_ARMS)
WORKSPACE_DEFAULTS = {"native_conv_workspace_reuse": "off", "native_conv_dx_batch_size": 1}
WORKSPACE_PROFILE_DEFAULTS = {**OPTION_DEFAULTS, **WORKSPACE_DEFAULTS, "packed_offload_selection_mode": "peak"}
WORKSPACE_PATTERN = "native_conv_private_workspace_tiled_dx"
OPTION_PROFILES[WORKSPACE_PROFILE] = WORKSPACE_PROFILE_DEFAULTS
OPTION_PROFILES[COMBINED_PROFILE] = WORKSPACE_PROFILE_DEFAULTS
OPTION_PROFILES[SELECTION_PROFILE] = WORKSPACE_PROFILE_DEFAULTS
OPTION_PROFILES[ARENA_PROFILE] = WORKSPACE_PROFILE_DEFAULTS
FUSION_PATTERN_PHASES[WORKSPACE_PATTERN] = "backward"


def option_enabled(option, value):
    if option == "native_conv_workspace_reuse":
        return value != "off"
    if option in FUSION_STAGE_OPTIONS.values():
        return value != "off"
    return value == "dw_first" if option == "native_conv_backward_order" else value is True


def _option_valid(option, value):
    if option == "native_conv_workspace_reuse":
        return type(value) is str and value in ("off", "tiled_dx", "whole_batch")
    if option == "native_conv_dx_batch_size":
        return type(value) is int and value > 0
    if option == "packed_offload_selection_mode":
        return type(value) is str and value in ("peak", "all_eligible", "regional_order")
    if option == "native_conv_backward_order":
        return type(value) is str and value in ("source", "dw_first")
    for flag, stage_option in FUSION_STAGE_OPTIONS.items():
        if option == stage_option:
            return type(value) is str and value in FUSION_STAGE_PATTERNS[flag]
    return option in OPTION_DEFAULTS and type(value) is bool


def _option_profile(options, declared=None):
    if declared is None:
        # Existing two/four-option manifests predate explicit profile names.
        declared = next((name for name, defaults in OPTION_PROFILES.items()
                         if name not in ("native_training_fusion_v3", *PAIRED_MEMORY_PROFILES) and set(options) == set(defaults)), None)
    if (type(declared) is not str or declared not in OPTION_PROFILES
            or set(options) != set(OPTION_PROFILES[declared])):
        raise ValueError("Experiment requires a supported exact typed option profile and explicit v3 profile marker")
    return declared


def _stage_contract_valid(options):
    return all((options.get(flag, False) is True) == (options.get(stage, "off") != "off")
               for flag, stage in FUSION_STAGE_OPTIONS.items())


def pass_selections(options, candidates):
    """Describe required stage executors; counts alone cannot satisfy a stage."""
    rows = {}
    contracts = {option: {"off": (), "on": (pattern,)} for option, pattern in PASS_PATTERNS.items()}
    contracts.update(FUSION_STAGE_PATTERNS)
    contracts["native_conv_workspace_reuse"] = {"off": (), "tiled_dx": (WORKSPACE_PATTERN,), "whole_batch": ()}
    for option, stages in contracts.items():
        value = options.get(option, {**OPTION_DEFAULTS, **WORKSPACE_DEFAULTS}[option])
        requested = option_enabled(option, value)
        stage_option = FUSION_STAGE_OPTIONS.get(option)
        stage = options.get(stage_option, "off") if stage_option else "on" if requested else "off"
        if option == "native_conv_workspace_reuse":
            stage = value
        required = stages.get(stage, ()) if requested and type(stage) is str else ()
        patterns = tuple(dict.fromkeys(pattern for values in stages.values() for pattern in values))
        counts = {pattern: sum(item.get("selected") is True and item.get("pattern") == pattern
                               and (pattern not in FUSION_PATTERN_PHASES
                                    or item.get("execution_phase") == FUSION_PATTERN_PHASES[pattern])
                               for item in candidates) for pattern in patterns}
        invalid_phases = tuple(pattern for pattern in patterns if pattern in FUSION_PATTERN_PHASES and any(
            item.get("selected") is True and item.get("pattern") == pattern
            and item.get("execution_phase") != FUSION_PATTERN_PHASES[pattern] for item in candidates))
        missing = tuple(pattern for pattern in required if not counts[pattern])
        unexpected = tuple(pattern for pattern in patterns if counts[pattern] and pattern not in required)
        rows[option] = {"requested": requested, "requested_value": value,
            "requested_stage": stage if stage_option or option == "native_conv_workspace_reuse" else None, "required_patterns": required,
            "selected_by_pattern": counts, "selected_regions": sum(counts.values()),
            "missing_patterns": missing, "unexpected_patterns": unexpected,
            "invalid_phase_patterns": invalid_phases,
            "selection_complete": requested and bool(required) and not missing and not unexpected and not invalid_phases}
    return rows


def parser():
    from examples.vae.benchmark_execution import parser as paired_parser

    result = paired_parser()
    result.description = __doc__
    result.set_defaults(trials=None, output=Path("work/vae_native_optimization.json"))
    result.add_argument("--experiment-manifest", type=Path, required=True)
    result.add_argument("--configuration-preflight-only", action="store_true",
        help="Validate pinned paired memory configurations without allocating a model or input")
    result.add_argument("--host-headroom-mib", type=int, default=8192,
        help="Reserve host/job headroom outside declared physical pinned capacity")
    return result


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _resolve(value, manifest):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def verify_arm(arm, manifest):
    """Verify wheel and installed package bytes without importing either wheel."""
    wheel = _resolve(arm["wheel"]["path"], manifest)
    if _sha(wheel.read_bytes()) != arm["wheel"]["sha256"]:
        raise ValueError("Experiment wheel checksum mismatch")
    target = _resolve(arm["install_root"], manifest)
    package = target / "deepopt"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Experiment wheel membership/CRC mismatch")
        metadata_name, = [name for name in names if name.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        expected = arm["expected_installed"]
        if metadata["Name"] != "deep-optimization" or metadata["Version"] != expected["version"]:
            raise ValueError("Experiment wheel metadata differs from its pinned installation")
        wheel_files = {name.removeprefix("deepopt/"): _sha(archive.read(name))
                       for name in names if name.startswith("deepopt/") and not name.endswith("/")}
    installed, digest = {}, hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if not path.resolve().is_relative_to(package.resolve()):
            raise ValueError("Experiment package resolves outside the pinned target")
        name, data = path.relative_to(package).as_posix(), path.read_bytes()
        installed[name] = _sha(data)
        digest.update(name.encode() + b"\0" + data + b"\0")
    if installed != wheel_files or digest.hexdigest() != expected["source_sha256"] or len(installed) != expected["files_hashed"]:
        raise ValueError("Installed experiment package differs from its wheel/source identity")
    return {"distribution_name": "deep-optimization", **expected, "package_path": str(package),
        "import_path": str(package / "__init__.py"), "install_root": str(target), "editable": False}


def load_manifest(path):
    from examples.vae.native_execution import _json_input

    manifest, source = _json_input(path)
    if manifest.get("experiment_profile") == WORKSPACE_PROFILE:
        return _workspace_manifest(manifest), source["sha256"]
    if manifest.get("experiment_profile") == COMBINED_PROFILE:
        return _combined_manifest(manifest), source["sha256"]
    if manifest.get("experiment_profile") == SELECTION_PROFILE:
        return _selection_manifest(manifest), source["sha256"]
    if manifest.get("experiment_profile") == ARENA_PROFILE:
        return _arena_manifest(manifest), source["sha256"]
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or type(manifest.get("arms")) is not dict or set(manifest["arms"]) != {"A", "B", "C"}):
        raise ValueError("Experiment requires schema1 and exactly the pinned A/B/C arms")
    arms = manifest["arms"]
    for arm in arms.values():
        if type(arm) is not dict or not {"wheel", "install_root", "expected_installed", "native_options"} <= arm.keys():
            raise ValueError("Experiment arm requires wheel, installation, identity and native options")
        expected, wheel = arm["expected_installed"], arm["wheel"]
        if (type(expected) is not dict or set(expected) != {"version", "source_sha256", "files_hashed"}
                or type(wheel) is not dict or set(wheel) != {"path", "sha256"}
                or type(arm["native_options"]) is not dict):
            raise ValueError("Experiment arm has an invalid identity/options schema")
        if (type(expected["version"]) is not str or not expected["version"].strip()
                or type(expected["files_hashed"]) is not int or expected["files_hashed"] < 1
                or any(type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None
                       for value in (expected["source_sha256"], wheel["sha256"]))
                or any(type(value) is not str or not value.strip()
                       for value in (wheel["path"], arm["install_root"]))):
            raise ValueError("Experiment arm requires nonempty paths, version and strict source hashes/count")
    if (arms["A"]["expected_installed"]["version"] != CONTROL_VERSION
            or arms["A"]["expected_installed"]["source_sha256"] != CONTROL_SOURCE_SHA
            or arms["A"]["wheel"]["sha256"] != CONTROL_WHEEL_SHA
            or arms["A"]["expected_installed"]["files_hashed"] != 457
            or arms["A"].get("native_options", {})):
        raise ValueError("A must be the unchanged .4 native control with no option overrides")
    if (arms["B"]["expected_installed"] != arms["C"]["expected_installed"]
            or arms["B"]["wheel"] != arms["C"]["wheel"]
            or _resolve(arms["B"]["install_root"], path) != _resolve(arms["C"]["install_root"], path)
            or arms["B"]["expected_installed"]["version"] == CONTROL_VERSION):
        raise ValueError("B and C must pin the same uniquely versioned candidate wheel and installation")
    if _resolve(arms["A"]["install_root"], path) == _resolve(arms["B"]["install_root"], path):
        raise ValueError("The historical control and candidate need separate installation targets")
    before, after = arms["B"].get("native_options", {}), arms["C"].get("native_options", {})
    profile = _option_profile(before, manifest.get("experiment_profile"))
    if (set(after) != set(before)
            or any(not _option_valid(key, value) or value != OPTION_PROFILES[profile][key] for key, value in before.items())
            or any(not _option_valid(key, value) for key, value in after.items())
            or not _stage_contract_valid(after)
            or not any(option_enabled(key, value) for key, value in after.items())):
        raise ValueError("B/C require a supported typed option profile: B at defaults, C with selected optimizations")
    if profile == "native_convolution_wave2" and any(after[key] for key in PASS_OPTIONS):
        raise ValueError("Convolution experiments keep the existing memory passes disabled")
    if profile == "native_training_fusion_v3" and any(after[key] != value for key, value in CONVOLUTION_DEFAULTS.items()):
        raise ValueError("Training fusion experiments keep every legacy/convolution option at its default")
    return manifest, source["sha256"]


def experiment_plan(args):
    from examples.vae import benchmark_execution as paired

    manifest, digest = load_manifest(args.experiment_manifest.resolve())
    args = deepcopy(args)
    args.trials = args.trials if args.trials is not None else 6 if manifest.get("experiment_profile") in PAIRED_MEMORY_PROFILES else 3
    if type(args.trials) is not int or args.trials < 1:
        raise ValueError("trials must be positive")
    if getattr(args, "arm_manifest", None) is not None:
        raise ValueError("A/B/C arm pins come only from the experiment manifest")
    if getattr(args, "memory_audit", True) is not True:
        raise ValueError("Pinned A/B/C comparisons require the worker memory audit")
    common = paired.benchmark_plan(args)
    if manifest.get("experiment_profile") in PAIRED_MEMORY_PROFILES:
        return _workspace_plan(args, common, manifest, digest)
    if getattr(args, "configuration_preflight_only", False):
        raise ValueError("--configuration-preflight-only requires a paired memory profile")
    identities = {name: verify_arm(arm, args.experiment_manifest.resolve()) for name, arm in manifest["arms"].items()}
    common.update(scope="pinned_native_compiler_A_B_C_experiment", manifest_sha256=digest,
        experiment_manifest=str(args.experiment_manifest.resolve()), arms=manifest["arms"],
        experiment_profile=_option_profile(manifest["arms"]["B"]["native_options"], manifest.get("experiment_profile")),
        installed_identities=identities,
        order=[list("ABC" if index % 2 == 0 else "CBA") for index in range(args.trials)],
        authorized_differences={"A_B": ["pinned_wheel_version_and_package_source"],
            "B_C": sorted(name for name in manifest["arms"]["C"]["native_options"]
                          if option_enabled(name, manifest["arms"]["C"]["native_options"][name]))},
        independent_samples="fresh process blocks; B is shared within each A/B/C block",
        performance_policy_promoted=False)
    common["base_native_plan"] = common.pop("workers")["native"]
    common["process_isolation"] = "fresh sequential native process per arm and block; one model on the GPU at a time"
    common["supervisor_sources"] = {str(path.relative_to(ROOT)): _sha(path.read_bytes())
        for path in (Path(__file__).resolve(), Path(paired.__file__).resolve(),
                     ROOT / "examples/vae/memory_comparison.py")}
    return common


def _identity(row, *, migration=False):
    from examples.vae.benchmark_execution import comparable_identity

    identity = deepcopy(comparable_identity(row))
    if migration:
        identity.pop("installed", None)
        provenance = identity.get("provenance")
        if isinstance(provenance, dict):
            for name in ("deepopt_source_sha256", "source_sha256", "deepopt_version", "deepopt_path"):
                provenance.pop(name, None)
    return identity


def compare_pair(left, right, expected_identities, *, migration=False):
    from examples.vae.benchmark_execution import _metrics, _worker_errors

    reasons = []
    for row in (left, right):
        reasons.extend(row["arm"] + ":" + reason for reason in _worker_errors(row))
        if row.get("installed") != expected_identities[row["arm"]]:
            reasons.append(row["arm"] + ":installation_does_not_match_authorized_arm")
        if row.get("settings") != row.get("expected_settings"):
            reasons.append(row["arm"] + ":settings_differ_from_declared_plan")
        if row["arm"] != "A":
            options = (row.get("settings") or {}).get("runtime", {}).get("compile", {}).get("options", {})
            compiler = row.get("compiler_report_summary", {})
            effective = compiler.get("effective_compiler_options", ())
            candidates = compiler.get("fusion_candidates", ())
            for option, default in {**OPTION_DEFAULTS, **WORKSPACE_DEFAULTS}.items():
                value = options.get(option, default)
                if (not _option_valid(option, value) or not effective
                        or any(option in options and option not in item
                            or type(item.get(option, default)) is not type(value)
                            or item.get(option, default) != value for item in effective)):
                    reasons.append(row["arm"] + ":effective_pass_option_mismatch:" + option)
            if not _stage_contract_valid(options):
                reasons.append(row["arm"] + ":invalid_fusion_flag_stage_contract")
            for option, selection in pass_selections(options, candidates).items():
                reasons.extend(row["arm"] + ":fusion_pattern_phase_mismatch:" + option + ":" + pattern
                               for pattern in selection["invalid_phase_patterns"])
                if not selection["requested"] and selection["selected_regions"]:
                    reasons.append(row["arm"] + ":pass_selected_while_disabled:" + option)
                elif selection["unexpected_patterns"]:
                    reasons.extend(row["arm"] + ":unexpected_stage_pattern_selected:" + option + ":" + pattern
                                   for pattern in selection["unexpected_patterns"])
                if selection["requested"] and str(row.get("device", "")).startswith("cuda"):
                    if selection["missing_patterns"]:
                        reasons.append(row["arm"] + ":requested_pass_not_selected:" + option)
                        reasons.extend(row["arm"] + ":requested_stage_pattern_not_selected:" + option + ":" + pattern
                                       for pattern in selection["missing_patterns"])
    a, b = _identity(left, migration=migration), _identity(right, migration=migration)
    reasons.extend("identity_mismatch:" + name for name in a if a[name] != b[name])
    ratios = {}
    if not reasons:
        lm, rm = _metrics(left), _metrics(right)
        ratios["runtime_speedup_left_over_right"] = lm["wall_ms_per_update"] / rm["wall_ms_per_update"]
        ratios["wall_time_reduction_percent"] = 100 * (1 - rm["wall_ms_per_update"] / lm["wall_ms_per_update"])
        for scope in ("steady", "startup", "total"):
            for kind in ("allocated", "reserved"):
                name = f"{scope}_peak_{kind}_bytes"
                if lm[name] is not None and rm[name] is not None:
                    ratios[f"{scope}_{kind}_reduction_percent"] = 100 * (lm[name] - rm[name]) / lm[name]
    return {"valid": not reasons, "invalid_reasons": reasons, "left": left["arm"],
            "right": right["arm"], "ratios": ratios}


def aggregate_blocks(results, identities, blocks, *, start_free_tolerance_bytes=256 * 2**20):
    summary = {"valid": False, "blocks": [], "invalid_reasons": [], "metrics": {},
        "ratios": {}, "independent_blocks": blocks, "performance_policy_promoted": False,
        "target_CUDA_evidence_complete": False, "pass_selections": []}
    from examples.vae.benchmark_execution import _metrics, _statistics
    from examples.vae.memory_comparison import compare_memory_baselines

    summary["memory_isolation"] = compare_memory_baselines(results, tolerance_bytes=start_free_tolerance_bytes)
    summary["invalid_reasons"].extend("memory_isolation:" + reason
                                     for reason in summary["memory_isolation"]["invalid_reasons"])

    anchors = {}
    if len(results) != 3 * blocks or any(row.get("block") not in range(1, blocks + 1) for row in results):
        summary["invalid_reasons"].append("unexpected_worker_count_or_block")
    for block in range(1, blocks + 1):
        rows = [row for row in results if row["block"] == block]
        arms = {name: [row for row in rows if row["arm"] == name] for name in "ABC"}
        if any(len(value) != 1 for value in arms.values()):
            summary["invalid_reasons"].append(f"block_{block}:incomplete_arms")
            continue
        a, b, c = (arms[name][0] for name in "ABC")
        for row in (a, b, c):
            candidates = row.get("compiler_report_summary", {}).get("fusion_candidates", ())
            options = (row.get("settings") or {}).get("runtime", {}).get("compile", {}).get("options", {})
            summary["pass_selections"].append({"block": block, "arm": row["arm"],
                "passes": pass_selections(options, candidates)})
        pairs = {"wheel_migration_A_B": compare_pair(a, b, identities, migration=True),
                 "optimization_B_C": compare_pair(b, c, identities)}
        summary["blocks"].append({"block": block, "comparisons": pairs})
        for label, pair in pairs.items():
            summary["invalid_reasons"].extend(f"block_{block}:{label}:" + error for error in pair["invalid_reasons"])
        for name, row in zip("ABC", (a, b, c)):
            identity = _identity(row)
            if name in anchors and identity != anchors[name]:
                summary["invalid_reasons"].append(f"block_{block}:{name}:identity_mismatch_across_blocks")
            anchors.setdefault(name, identity)
    summary["valid"] = len(summary["blocks"]) == blocks and not summary["invalid_reasons"]
    for arm in "ABC":
        from examples.vae.benchmark_execution import _worker_errors

        rows = [row for row in results if row["arm"] == arm and not _worker_errors(row)
                and row.get("installed") == identities[arm] and row.get("settings") == row.get("expected_settings")]
        names = set().union(*(_metrics(row) for row in rows)) if rows else set()
        summary["metrics"][arm] = {name: _statistics([_metrics(row)[name] for row in rows]) for name in sorted(names)}
    if summary["valid"]:
        for label in ("wheel_migration_A_B", "optimization_B_C"):
            rows = [block["comparisons"][label]["ratios"] for block in summary["blocks"]]
            names = set().union(*(row.keys() for row in rows))
            summary["ratios"][label] = {name: _statistics([row[name] for row in rows if name in row]) for name in sorted(names)}
        summary["target_CUDA_evidence_complete"] = (blocks >= 3
            and all(row.get("physical_target_executed") is True for row in results)
            and all(not value["requested"] or value["selection_complete"]
                    for row in summary["pass_selections"] if row["arm"] == "C"
                    for value in row["passes"].values()))
    return summary


def _memory_arm_contracts(value, names):
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or type(value.get("arms")) is not dict or set(value["arms"]) != set(names)):
        raise ValueError("Paired memory experiments require schema1 and exactly " + "/".join(names))
    arms = value["arms"]
    for arm in arms.values():
        if type(arm) is not dict or set(arm) != {"wheel", "install_root", "expected_installed", "native_options"}:
            raise ValueError("Workspace arm must pin its wheel, installation, identity and native options")
        expected, wheel, options = arm["expected_installed"], arm["wheel"], arm["native_options"]
        if (type(expected) is not dict or set(expected) != {"version", "source_sha256", "files_hashed"}
                or type(wheel) is not dict or set(wheel) != {"path", "sha256"}
                or type(options) is not dict or set(options) != set(WORKSPACE_PROFILE_DEFAULTS)
                or type(expected["version"]) is not str or not expected["version"].strip()
                or type(expected["files_hashed"]) is not int or expected["files_hashed"] < 1
                or any(type(part) is not str or not part.strip() for part in (wheel["path"], arm["install_root"]))
                or any(type(part) is not str or re.fullmatch("[0-9a-f]{64}", part) is None
                       for part in (wheel["sha256"], expected["source_sha256"]))
                or any(not _option_valid(key, part) for key, part in options.items())):
            raise ValueError("Workspace arm has invalid strict identity or exact11 typed options")
        if any(options[key] != default for key, default in OPTION_DEFAULTS.items()):
            raise ValueError("Workspace experiments keep all eight earlier optimization controls at defaults")
    before, after = (arms[name] for name in names)
    if any(before[key] != after[key] for key in ("wheel", "install_root", "expected_installed")):
        raise ValueError("Paired memory arms must share one candidate wheel and installation")
    return before, after


def _workspace_manifest(value):
    """A same-release, fixed-residency ablation has no historical A arm."""
    from examples.vae.benchmark_packed_offload import normalize_policy

    before, after = _memory_arm_contracts(value, ("D0", "D1"))
    if (before["native_options"]["native_conv_workspace_reuse"] != "off"
            or after["native_options"]["native_conv_workspace_reuse"] != "tiled_dx"
            or before["native_options"]["packed_offload_selection_mode"] not in ("peak", "all_eligible")
            or any(before["native_options"][key] != after["native_options"][key]
                   for key in WORKSPACE_PROFILE_DEFAULTS if key != "native_conv_workspace_reuse")):
        raise ValueError("D0/D1 must share one candidate and differ only off/tiled_dx; whole_batch is unqualified")
    policy = normalize_policy(value.get("packed_activation_offload"))
    if policy.get("version") != 2 or policy["enabled"] is not True or policy["policy"] != "demand":
        raise ValueError("Workspace experiments require one enabled version2 demand policy")
    return {**value, "packed_activation_offload": policy}


def _combined_manifest(value):
    from examples.vae.benchmark_packed_offload import normalize_policy

    before, after = _memory_arm_contracts(value, ("R0", "BEST"))
    if (before["native_options"]["native_conv_workspace_reuse"] != "off"
            or after["native_options"]["native_conv_workspace_reuse"] not in ("off", "tiled_dx")
            or before["native_options"]["packed_offload_selection_mode"] not in ("peak", "all_eligible")
            or any(before["native_options"][key] != after["native_options"][key]
                   for key in WORKSPACE_PROFILE_DEFAULTS if key != "native_conv_workspace_reuse")):
        raise ValueError("Combined arms allow only the declared off/tiled_dx workspace difference")
    if type(value.get("policies")) is not dict or set(value["policies"]) != {"R0", "BEST"}:
        raise ValueError("Combined comparison requires explicit R0/BEST policies")
    policies = {name: normalize_policy(part) for name, part in value["policies"].items()}
    if any(policy.get("version") != 2 or policy["policy"] != "demand" for policy in policies.values()):
        raise ValueError("Combined v2 supports the chosen demand candidate; predictive selection remains unqualified")
    if policies["R0"]["enabled"] is not False or policies["BEST"]["enabled"] is not True:
        raise ValueError("Combined R0 must disable offload and BEST must enable demand")
    if ({key: part for key, part in policies["R0"].items() if key != "enabled"}
            != {key: part for key, part in policies["BEST"].items() if key != "enabled"}):
        raise ValueError("Combined arms must keep every declared physical budget and admission control identical")
    return {**value, "policies": policies}


def _selection_manifest(value):
    from examples.vae.benchmark_packed_offload import normalize_policy

    before, after = _memory_arm_contracts(value, ("S0", "S1"))
    if (before["native_options"]["packed_offload_selection_mode"] != "regional_order"
            or after["native_options"]["packed_offload_selection_mode"] != "peak"
            or before["native_options"]["native_conv_workspace_reuse"] not in ("off", "tiled_dx")
            or any(before["native_options"][key] != after["native_options"][key]
                   for key in WORKSPACE_PROFILE_DEFAULTS if key != "packed_offload_selection_mode")):
        raise ValueError("S0/S1 allow only the diagnostic regional_order/peak selection difference")
    policy = normalize_policy(value.get("packed_activation_offload"))
    if policy.get("version") != 2 or policy["enabled"] is not True or policy["policy"] != "demand":
        raise ValueError("Selection experiments require one enabled version2 demand policy and arena")
    return {**value, "packed_activation_offload": policy}


def _arena_manifest(value):
    from examples.vae.benchmark_packed_offload import normalize_policy

    before, after = _memory_arm_contracts(value, ("H0", "H1"))
    if (before["native_options"] != after["native_options"]
            or before["native_options"]["packed_offload_selection_mode"] != "peak"
            or before["native_options"]["native_conv_workspace_reuse"] not in ("off", "tiled_dx")):
        raise ValueError("Arena arms require identical math and peak-only selection controls")
    if type(value.get("policies")) is not dict or set(value["policies"]) != {"H0", "H1"}:
        raise ValueError("Arena comparison requires explicit H0/H1 policies")
    policies = {name: normalize_policy(part) for name, part in value["policies"].items()}
    if (policies["H0"].get("version", 1) != 1 or policies["H1"].get("version") != 2
            or any(part["enabled"] is not True or part["policy"] != "demand" or part["host_pageable_bytes"] != 0
                   for part in policies.values())
            or {key: part for key, part in policies["H0"].items() if key != "version"}
                != {key: part for key, part in policies["H1"].items() if key != "version"}):
        raise ValueError("Arena comparison permits only demand policy version1/version2 at identical explicit pinned budgets")
    return {**value, "policies": policies}


def _workspace_plan(args, common, manifest, digest):
    from examples.vae import benchmark_packed_offload as offload, native_execution

    if args.trials != 1 and (args.trials < 6 or args.trials % 2):
        raise ValueError("Workspace experiments require six or more balanced pairs, or explicit --trials 1 triage")
    if type(getattr(args, "host_headroom_mib", 8192)) is not int or getattr(args, "host_headroom_mib", 8192) < 0:
        raise ValueError("host-headroom-mib must be a nonnegative integer")
    if args.profile_after_measurement or args.profile_memory or args.profile_record_shapes or args.profile_updates != 1:
        raise ValueError("Workspace clean pairs exclude profiler updates; run separate diagnostic workers")
    if args.device == "cpu" and not (args.dry_run or getattr(args, "configuration_preflight_only", False)):
        raise ValueError("Enabled packed offload requires CUDA; a CPU run cannot substitute for the workspace benchmark")
    profile = manifest["experiment_profile"]
    workspace, arena, selection = profile == WORKSPACE_PROFILE, profile == ARENA_PROFILE, profile == SELECTION_PROFILE
    names = MEMORY_ARMS[profile]
    identities = {name: verify_arm(arm, args.experiment_manifest.resolve()) for name, arm in manifest["arms"].items()}
    if identities[names[0]] != identities[names[1]]:
        raise ValueError("Workspace arms must resolve to the same candidate installation")
    identity, base = identities[names[0]], common["workers"]["native"]
    policies = ({name: deepcopy(manifest["packed_activation_offload"]) for name in names}
                if workspace or selection else manifest["policies"])
    workers = {}
    for name, spec in manifest["arms"].items():
        worker = deepcopy(base)
        worker["settings"] = offload._apply_policy(worker["settings"], policies[name])
        worker["settings"]["runtime"]["compile"]["options"].update(spec["native_options"])
        worker["expected_installed_version"] = identity["version"]
        worker["resolved_native_options_sha256"] = native_execution.resolved_native_options_sha256(worker["settings"])
        worker["host_headroom_mib"] = args.host_headroom_mib
        worker["host_headroom_bytes"] = args.host_headroom_mib * 2**20
        worker["host_capacity_policy"] = {"reserved_headroom_bytes": args.host_headroom_mib * 2**20,
            "headroom_mib": args.host_headroom_mib, "read_only_preallocation_check": True, "requested_capacity_clamped": False}
        workers[name] = worker
    candidate = {key: deepcopy(manifest["arms"][names[0]][key]) for key in ("wheel", "install_root", "expected_installed")}
    scopes = {WORKSPACE_PROFILE: "fixed_demand_native_convolution_workspace_ablation",
        COMBINED_PROFILE: "resident_to_chosen_combined_demand_candidate",
        SELECTION_PROFILE: "fixed_v2_arena_demand_selection_ablation", ARENA_PROFILE: "fixed_history_pinned_pool_layout_ablation"}
    differences = (["native_conv_workspace_reuse:off_to_tiled_dx"] if workspace else
        ["packed_offload_selection_mode:regional_order_to_peak"] if selection else
        ["packed_activation_offload.version:1_to_2"] if arena else
        ["packed_activation_offload.enabled:False_to_True"] +
        (["native_conv_workspace_reuse:off_to_tiled_dx"] if manifest["arms"]["BEST"]["native_options"]["native_conv_workspace_reuse"] == "tiled_dx" else []))
    return {**common, "scope": scopes[profile],
        "experiment_profile": manifest["experiment_profile"], "experiment_manifest": str(args.experiment_manifest.resolve()),
        "manifest_sha256": digest, "manifest_source": {"path": str(args.experiment_manifest.resolve()), "sha256": digest},
        "arms": manifest["arms"], "manifest": {**manifest, "candidate": candidate, "policies": policies},
        "installed": identity, "installed_identities": identities, "workers": workers,
        "active_arms": list(names), "order": [list(names if index % 2 == 0 else reversed(names))
                                                   for index in range(args.trials)],
        "source_sha256": _memory_source_identity(), "trials_requested": args.trials,
        "host_headroom_mib": args.host_headroom_mib, "host_headroom_bytes": args.host_headroom_mib * 2**20,
        "baseline_tolerance_bytes": args.start_free_tolerance_mib * 2**20,
        "configuration_preflight_workers": 2, "independent_samples": "fresh balanced process pairs; updates are not independent trials",
        "authorized_differences": differences,
        "residency_control": {"policy": "demand", "selection_mode": manifest["arms"][names[0]]["native_options"]["packed_offload_selection_mode"],
            "eligible_inventory_must_match": workspace or selection,
            "selected_and_actual_admission_must_match": workspace or arena,
            "candidate_selected_and_actual_admission_must_match": workspace,
            "all_eligible_requires_compiled_capacity_check": True,
            "drift_action": "retain_descriptive_points_and_suppress_independent_ratios"},
        "performance_policy_promoted": False, "target_CUDA_evidence_complete": False,
        "chosen_candidate_scope": "BEST is the declared demand comparison point, not a globally optimal claim",
        "selection_reference_scope": "regional_order is a v2 diagnostic greedy heuristic, not an exact historical .3 replay",
        "arena_physical_check_scope": "H0 recorded approximate Torch process pinned allocated peak includes input/cache and must fit; this does not claim a v1 allocator cap",
        "legacy_host_capacity_preflight": {"required_bytes": 2 * policies[names[0]]["host_pinned_bytes"] if arena else None,
            "scope": "conservative_legacy_rounding_envelope_plus_headroom_not_a_v1_physical_cap",
            "before_model_allocation": True},
        "milestone_estimator": {"memory": "observed_mean_paired_whole_run_allocated_peak_reduction_percent_at_least_10",
            "timing": "one_sided_95_percent_upper_bound_on_mean_paired_log_BEST_over_R0_runtime_at_most_1.01",
            "minimum_independent_pairs": 6, "critical_value": 2.0150483733330233,
            "critical_value_scope": "Student_t_0.95_df5_conservative_for_at_least_six_pairs",
            "assumption": "independent_balanced_pairs_with_approximately_normal_paired_log_runtime_ratios",
            "not_automatic_performance_policy_promotion": True},
        "process_isolation": "two configuration-only processes, then one fresh sequential CUDA process per arm and pair"}


def _memory_source_identity():
    from examples.vae.benchmark_packed_offload import source_identity

    return {**source_identity(), **{name: _sha((ROOT / name).read_bytes()) for name in
        ("examples/vae/execution_inventory.py", "benchmarks/compiler/installed_runtime.py")}}


def _workspace_identity(row):
    from examples.vae.benchmark_packed_offload import _identity as transport_identity

    result = deepcopy(transport_identity(row))
    options = result.get("native_math_options")
    if isinstance(options, dict):
        options.pop("native_conv_workspace_reuse", None)
        # This is a separately pinned residency axis, never an implicit math
        # difference. D0/D1 still require the same value in their manifests.
        options.pop("packed_offload_selection_mode", None)
    return result


def _workspace_component(value):
    fields = ("name", "saved_index", "shape", "dtype", "stride", "device", "nbytes", "role")
    if type(value) is not dict or any(key not in value for key in fields):
        raise ValueError("Missing packed component ABI")
    shape, stride = value["shape"], value["stride"]
    if (any(type(value[key]) is not str or not value[key] for key in ("name", "dtype", "device", "role"))
            or any(type(value[key]) is not int or value[key] < 0 for key in ("saved_index", "nbytes"))
            or not isinstance(shape, (list, tuple)) or not isinstance(stride, (list, tuple))
            or len(shape) != len(stride) or any(type(n) is not int or n < 0 for n in (*shape, *stride))):
        raise ValueError("Malformed packed component geometry")
    return {key: list(value[key]) if key in ("shape", "stride") else value[key] for key in fields}


def _workspace_history_from_report(compiler, *, require_candidate=True):
    """Use component ABI, never backward scheduling IDs, as residency identity."""
    from examples.vae.execution_inventory import _declared_native_regions

    owners, errors = _declared_native_regions(compiler)
    if errors:
        raise ValueError("Invalid declared native-region inventory: " + json.dumps(errors, sort_keys=True))
    regions, eligible = {}, set()
    for path, owner in owners:
        if "packed_activation_offload" not in owner:
            continue  # Disabled and legacy regions can be valid nonowners.
        envelope = owner["packed_activation_offload"]
        if not isinstance(envelope, dict) or not isinstance(envelope.get("plan"), dict):
            raise ValueError("Missing declared regional packed plan: " + path)
        plan = envelope["plan"]
        fw = plan.get("forward_signature")
        if type(fw) is not str or not fw or fw in regions:
            raise ValueError("Missing or ambiguous forward-region inventory identity")
        bundles = []
        for bundle in plan.get("bundles", ()):
            if type(bundle) is not dict or type(bundle.get("bundle_id")) is not str or not bundle["bundle_id"]:
                raise ValueError("Malformed selected packed bundle")
            components = [_workspace_component(part) for part in bundle.get("components", ())]
            if len(components) != 3 or len({part["name"] for part in components}) != 3:
                raise ValueError("Packed inventory must contain three distinct component leaves")
            size = sum(part["nbytes"] for part in components)
            if type(bundle.get("bytes")) is not int or bundle["bytes"] != size:
                raise ValueError("Packed bundle byte count contradicts component ABI")
            bundles.append({"bundle_id": bundle["bundle_id"], "bytes": size,
                "components": sorted(components, key=lambda part: part["name"])})
        if len({part["bundle_id"] for part in bundles}) != len(bundles):
            raise ValueError("Duplicate packed bundle identity")
        size = sum(part["bytes"] for part in bundles)
        if type(plan.get("selected_bytes")) is not int or plan["selected_bytes"] != size:
            raise ValueError("Regional selected bytes contradict packed components")
        regions[fw] = {"forward_signature": fw, "selected_bytes": size,
            "bundles": sorted(bundles, key=lambda part: part["bundle_id"])}
        scope = envelope.get("model_scope", {})
        inventory = scope.get("inventory") if isinstance(scope, dict) else None
        if isinstance(inventory, dict) or require_candidate:
            candidate = inventory.get("eligible_inventory_sha256") if isinstance(inventory, dict) else None
            if type(candidate) is not str or re.fullmatch("[0-9a-f]{64}", candidate) is None:
                raise ValueError("Missing canonical eligible component inventory hash")
            eligible.add(candidate)
    if not regions or len(eligible) > 1 or require_candidate and len(eligible) != 1:
        raise ValueError("Incomplete or ambiguous model-scope packed inventory")
    return {"schema_version": 1, "eligible_inventory_sha256": next(iter(eligible), None),
        "selected_regions": [regions[key] for key in sorted(regions)],
        "scope": "static_component_geometry_not_measured_unique_storage_or_GPU_peak"}


def _workspace_read_history(row, result_path):
    """Only a verified saved compiler artifact can authorize residency parity."""
    from examples.vae.benchmark_execution import _reject_json_constant

    artifact = row.get("compiler_report_artifact")
    if (type(artifact) is not dict or type(artifact.get("path")) is not str
            or type(artifact.get("bytes")) is not int or not 0 < artifact["bytes"] <= 256 * 2**20
            or type(artifact.get("sha256")) is not str or re.fullmatch("[0-9a-f]{64}", artifact["sha256"]) is None):
        raise ValueError("Missing bounded hashed full compiler report")
    path = Path(artifact["path"])
    path = path if path.is_absolute() else result_path.parent / path
    if not path.resolve().is_relative_to(result_path.parent.resolve()):
        raise ValueError("Compiler report is outside the worker artifact tree")
    with path.open("rb") as stream:
        data = stream.read(256 * 2**20 + 1)
    if len(data) != artifact["bytes"] or _sha(data) != artifact["sha256"]:
        raise ValueError("Full compiler report bytes changed")
    compiler = json.loads(data, parse_constant=_reject_json_constant)
    policy = row.get("settings", {}).get("runtime", {}).get("packed_activation_offload", {})
    return {**_workspace_history_from_report(compiler, require_candidate=policy.get("version", 1) == 2),
            "verified_source": deepcopy(artifact)}


def _workspace_admissions(row):
    from examples.vae import benchmark_packed_offload as offload

    offload._actual_admissions(row)  # Check untruncated frozen prefix first.
    static = {part["forward_signature"]: part for part in row["_workspace_history"]["selected_regions"]}
    result = []
    for receipt, expected_updates in ((row["packed_offload_profile"], row["warmup"]),
                                       (row["packed_offload_runtime"], row["warmup"] + row["updates"])):
        _, managers = offload._runtime({"packed_offload_runtime": receipt})
        sequences = {}
        for manager in managers:
            plans = manager.get("plans", ())
            regions = {}
            for plan in plans:
                fw = plan.get("forward_signature")
                if fw not in static or plan.get("region_id") in regions or fw in sequences:
                    raise ValueError("Runtime regional selection has no unique compiler inventory")
                expected = static[fw]
                ids = sorted(part["bundle_id"] for part in expected["bundles"])
                if (sorted(plan.get("selected_bundle_ids", ())) != ids
                        or plan.get("selected_bytes") != expected["selected_bytes"]):
                    raise ValueError("Runtime selected inventory contradicts the verified compiler report")
                regions[plan["region_id"]] = fw
                sequences[fw] = []
            for event in manager.get("admission_history", ()):
                fw = regions[event["region_id"]]
                sequences[fw].append({"resident_fallback": event["resident_fallback"],
                    "selected_bundle_ids": sorted(event["selected_bundle_ids"])})
        expected_regions = {fw for fw, part in static.items() if part["selected_bytes"]}
        if not expected_regions <= sequences.keys() or any(len(sequence) != expected_updates for sequence in sequences.values()):
            raise ValueError("Actual admission does not cover each logical forward exactly once")
        result.append({fw: sequences[fw] for fw in sorted(sequences)})
    return {"warmup": result[0], "complete": result[1]}


def _workspace_worker_errors(row, plan, arm):
    from examples.vae import benchmark_packed_offload as offload, native_execution

    reasons = offload.worker_errors(row, plan["manifest"]["policies"][arm], plan["workers"][arm]["settings"], plan["installed"])
    if not isinstance(row, dict):
        return reasons
    if row.get("origin") != plan["origin"]:
        reasons.append("workspace_origin_changed")
    expected = plan["workers"][arm]
    if any(row.get(name) != expected.get(name) for name in ("warmup", "updates", "requested_input_shape", "device",
            "execution_mode", "optimizer_factory", "model_factory", "input_mode", "gradient_mode", "host_capacity_policy")):
        reasons.append("workspace_execution_contract_differs_from_plan")
    inputs = row.get("input")
    shape = expected["requested_input_shape"]
    stride, elements = [], 1
    for size in reversed(shape):
        stride.insert(0, elements)
        elements *= size
    if (not isinstance(inputs, dict) or inputs.get("shape") != shape or inputs.get("stride") != stride
            or inputs.get("dtype") != "torch.float32" or inputs.get("contiguous") is not True
            or inputs.get("requires_grad") is not False or inputs.get("storage_bytes") != elements * 4):
        reasons.append("physical_dense_input_contract_unverified")
    options = plan["workers"][arm]["settings"]["runtime"]["compile"]["options"]
    compiler = row.get("compiler_report_summary")
    compiler = compiler if isinstance(compiler, dict) else {}
    if (type(compiler.get("native_autodiff_regions")) is not int or compiler["native_autodiff_regions"] < 1
            or type(compiler.get("native_autodiff_operations")) is not int or compiler["native_autodiff_operations"] < 1
            or compiler.get("runtime_fallbacks") != 0 or compiler.get("legacy_backward_operations") not in ([], ())):
        reasons.append("full_native_backward_coverage_unverified")
    measurement = row.get("measurement")
    if (not isinstance(measurement, dict) or type(measurement.get("native_graph_count_before")) is not int
            or measurement["native_graph_count_before"] < 1
            or measurement.get("native_graph_count_after") != measurement["native_graph_count_before"]):
        reasons.append("native_graph_reuse_unverified")
    effective, candidates = compiler.get("effective_compiler_options"), compiler.get("fusion_candidates")
    if not isinstance(effective, (list, tuple)) or not effective or any(not isinstance(item, dict) for item in effective):
        reasons.append("effective_workspace_options_unavailable")
    else:
        for name in WORKSPACE_PROFILE_DEFAULTS:
            if any(name not in item or type(item[name]) is not type(options[name]) or item[name] != options[name] for item in effective):
                reasons.append("effective_workspace_option_mismatch:" + name)
    if not isinstance(candidates, (list, tuple)) or any(not isinstance(item, dict) for item in candidates):
        reasons.append("workspace_candidates_unavailable")
    else:
        selections = pass_selections(options, candidates)
        for name, selection in selections.items():
            if selection["invalid_phase_patterns"] or selection["unexpected_patterns"]:
                reasons.append("unexpected_workspace_pattern:" + name)
            if (not selection["requested"] and selection["selected_regions"]
                    or selection["requested"] and not selection["selection_complete"]):
                reasons.append("workspace_pass_selection_mismatch:" + name)
    steps = row.get("steps")
    checked = [item for item in steps if isinstance(item, dict) and item.get("stage") == "warmup"] if isinstance(steps, list) else []
    if (len(checked) != row.get("warmup") or any(any(item.get(name) is not True for name in
            ("forward_completed", "backward_completed", "optimizer_completed", "all_trainable_gradients_present", "fp32_gradients", "finite_gradients"))
            for item in checked)):
        reasons.append("untimed_full_parameter_gradient_checks_incomplete")
    pin = row.get("_expected_arm_pin")
    if not isinstance(pin, dict) or row.get("experiment_arm", {}).get("manifest") != pin:
        reasons.append("workspace_arm_pin_not_verified")
    else:
        if (pin.get("arm") != arm or pin.get("expected_installed") != plan["manifest"]["candidate"]["expected_installed"]
                or pin.get("wheel_sha256") != plan["manifest"]["candidate"]["wheel"]["sha256"]
                or row.get("native_options_override", {}).get("sha256") != pin.get("native_options_sha256")
                or row.get("packed_offload_override", {}).get("sha256") != pin.get("packed_offload_sha256")):
            reasons.append("workspace_arm_input_pin_mismatch")
    expected_hash = native_execution.resolved_native_options_sha256(plan["workers"][arm]["settings"])
    if (row.get("resolved_native_options_sha256") != expected_hash
            or isinstance(pin, dict) and pin.get("resolved_native_options_sha256") != expected_hash):
        reasons.append("resolved_workspace_options_hash_mismatch")
    enabled = plan["manifest"]["policies"][arm]["enabled"]
    if enabled:
        try:
            if not isinstance(row.get("_workspace_history"), dict) or not row["_workspace_history"].get("verified_source"):
                raise ValueError(row.get("_workspace_history_error", "Verified packed inventory unavailable"))
            _workspace_admissions(row)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            reasons.append("workspace_history_unverified:" + str(error))
    policy = plan["manifest"]["policies"][arm]
    if policy.get("version", 1) == 2:
        capacity = row.get("host_capacity_preflight")
        required_capacity = policy["host_pinned_bytes"] + policy["host_pageable_bytes"] if enabled else 0
        if (not isinstance(capacity, dict) or capacity.get("admitted") is not True
                or capacity.get("required_physical_capacity_bytes") != required_capacity
                or capacity.get("reserved_headroom_bytes") != plan["host_headroom_bytes"]
                or capacity.get("requested_capacity_clamped") is not False):
            reasons.append("physical_host_resource_preflight_unverified")
    if enabled and policy.get("version", 1) == 2 and options["packed_offload_selection_mode"] == "all_eligible":
        inventory = row.get("packed_offload_inventory")
        policy = plan["manifest"]["policies"][arm]
        capacity = inventory.get("all_eligible_host_physical_capacity_bytes") if isinstance(inventory, dict) else None
        if (not isinstance(inventory, dict) or inventory.get("status") != "available"
                or inventory.get("selection_mode") != "all_eligible" or inventory.get("all_eligible_selected") is not True
                or type(capacity) is not int or capacity < 0
                or capacity > (policy["host_pageable_bytes"] or policy["host_pinned_bytes"])):
            reasons.append("all_eligible_capacity_or_coverage_unverified")
    if plan["experiment_profile"] == ARENA_PROFILE and arm == "H0":
        check = _legacy_pinned_check(row, policy)
        if not check["valid"]:
            reasons.append(check["reason"])
    return list(dict.fromkeys(reasons))


def _legacy_pinned_check(row, policy):
    """Recorded whole-process cost is conservative, not a v1 hard-cap proof."""
    snapshot = row
    for key in ("memory_audit", "snapshots", "after_postchecks", "pinned_host_allocator"):
        snapshot = snapshot.get(key, {}) if isinstance(snapshot, dict) else {}
    stats = snapshot.get("statistics", {}) if isinstance(snapshot, dict) else {}
    peak, current = stats.get("allocated_bytes.peak"), stats.get("allocated_bytes.current")
    available = (isinstance(snapshot, dict) and snapshot.get("status") == "available" and type(peak) is int and peak >= 0
                 and type(current) is int and current >= 0)
    observed = max(peak, current) if available else None
    valid = available and observed <= policy["host_pinned_bytes"]
    return {"valid": valid, "observed_process_pinned_allocated_peak_bytes": observed,
        "declared_physical_comparison_ceiling_bytes": policy["host_pinned_bytes"],
        "input_and_allocator_cache_included": True, "peak_is_approximate": True,
        "legacy_allocator_enforces_cap": False, "source_stage": "after_postchecks",
        "reason": None if valid else "legacy_process_pinned_peak_exceeds_ceiling" if available else "legacy_process_pinned_peak_unavailable"}


def aggregate_workspace(results, plan):
    """Retain each valid point; only identical history authorizes the ratio."""
    from examples.vae import benchmark_execution as paired, benchmark_packed_offload as offload
    from examples.vae.memory_comparison import compare_memory_baselines

    profile = plan["experiment_profile"]
    workspace, arena, selection = profile == WORKSPACE_PROFILE, profile == ARENA_PROFILE, profile == SELECTION_PROFILE
    names = tuple(plan["active_arms"])
    summary = {"valid": False, "invalid_reasons": [], "arms": {}, "blocks": [], "ratios": {},
        "independent_dense_reuse_attribution": False, "combined_point_exploratory": False,
        "independent_pinned_pool_layout_attribution": False, "independent_selection_attribution": False,
        "total_combined_improvement_attribution": False,
        "performance_policy_promoted": False, "target_CUDA_evidence_complete": False,
        "correctness_assessed": False, "numerical_comparisons_performed": False}
    summary["memory_isolation"] = compare_memory_baselines(
        [{**row, "arm": row.get("offload_arm", row.get("arm", "worker"))} for row in results],
        tolerance_bytes=plan["baseline_tolerance_bytes"])
    summary["invalid_reasons"] += ["memory_isolation:" + reason for reason in summary["memory_isolation"]["invalid_reasons"]]
    if arena:
        summary["arena_physical_checks"] = [{"block": row.get("block"), "arm": "H0",
            **_legacy_pinned_check(row, plan["manifest"]["policies"]["H0"])}
            for row in results if row.get("offload_arm") == "H0"]
    blocks = len(plan["order"])
    if len(results) != 2 * blocks or any(row.get("block") not in range(1, blocks + 1) for row in results):
        summary["invalid_reasons"].append("unexpected_workspace_worker_count_or_block")
    errors = {}
    for row in results:
        arm = row.get("offload_arm")
        try:
            errors[id(row)] = _workspace_worker_errors(row, plan, arm) if arm in names else ["unknown_memory_arm"]
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            errors[id(row)] = ["malformed_worker_receipt:" + type(error).__name__ + ":" + str(error)]
    for arm in names:
        rows = [row for row in results if row.get("offload_arm") == arm]
        good = [row for row in rows if not errors[id(row)]]
        metrics = [offload._metrics(row) for row in good]
        metric_names = set().union(*(item.keys() for item in metrics)) if metrics else set()
        summary["arms"][arm] = {"successful_trials": len(good), "total_trials": len(rows),
            "metrics": {name: paired._statistics([item.get(name) for item in metrics]) for name in sorted(metric_names)}}
    anchor, history_anchors = None, {}
    for block in range(1, blocks + 1):
        rows = [row for row in results if row.get("block") == block]
        by_arm = {arm: [row for row in rows if row.get("offload_arm") == arm] for arm in names}
        reasons, ratios, deltas = [], {}, {}
        if len(rows) != 2 or any(len(value) != 1 for value in by_arm.values()):
            reasons.append("missing_or_duplicate_workspace_arms")
        else:
            left, right = by_arm[names[0]][0], by_arm[names[1]][0]
            reasons += [row["offload_arm"] + ":" + reason for row in (left, right) for reason in errors[id(row)]]
            if not reasons:
                identity = _workspace_identity(left)
                if identity != _workspace_identity(right):
                    reasons.append("math_state_input_RNG_source_or_GPU_drift")
                if anchor is not None and anchor != identity:
                    reasons.append("identity_drift_across_workspace_pairs")
                anchor = identity if anchor is None else anchor
                for row in (left, right):
                    arm = row["offload_arm"]
                    if plan["manifest"]["policies"][arm]["enabled"]:
                        history = row["_workspace_history"]
                        canonical = (history["eligible_inventory_sha256"], history["selected_regions"], _workspace_admissions(row))
                        if arm in history_anchors and history_anchors[arm] != canonical:
                            reasons.append(arm + ":history_drift_across_pairs")
                        history_anchors.setdefault(arm, canonical)
                if workspace or arena or selection:
                    lh, rh = left["_workspace_history"], right["_workspace_history"]
                    if (workspace or selection) and lh["eligible_inventory_sha256"] != rh["eligible_inventory_sha256"]:
                        reasons.append("eligible_component_inventory_drift")
                    if (workspace or arena) and lh["selected_regions"] != rh["selected_regions"]:
                        reasons.append("selected_component_geometry_or_bytes_drift")
                    if (workspace or arena) and _workspace_admissions(left) != _workspace_admissions(right):
                        reasons.append("actual_admitted_history_drift")
                    if any(reason.endswith("_drift") for reason in reasons if "inventory" in reason or "component" in reason or "admitted" in reason):
                        summary["combined_point_exploratory"] = True
            if not reasons:
                lm, rm = offload._metrics(left), offload._metrics(right)
                ratios[f"runtime_speedup_{names[0]}_over_{names[1]}"] = lm["wall_ms_per_update"] / rm["wall_ms_per_update"]
                ratios[f"runtime_ratio_{names[1]}_over_{names[0]}"] = rm["wall_ms_per_update"] / lm["wall_ms_per_update"]
                for name in lm.keys() & rm.keys():
                    if paired._number(lm[name]) and paired._number(rm[name]):
                        deltas[name] = rm[name] - lm[name]
                for scope in ("steady", "startup", "total"):
                    for kind in ("allocated", "reserved"):
                        name = f"{scope}_peak_{kind}_bytes"
                        if paired._number(lm[name], positive=True) and paired._number(rm[name]):
                            ratios[f"{scope}_{kind}_reduction_percent"] = 100 * (lm[name] - rm[name]) / lm[name]
        summary["blocks"].append({"block": block, "valid": not reasons, "invalid_reasons": reasons,
                                  "ratios": ratios, "deltas_right_minus_left": deltas})
        summary["invalid_reasons"] += [f"block_{block}:" + reason for reason in reasons]
    summary["valid"] = bool(results) and not summary["invalid_reasons"]
    positions = {arm: [sum(order[index] == arm for order in plan["order"]) for index in (0, 1)] for arm in names}
    balanced = blocks >= 6 and all(len(set(value)) == 1 for value in positions.values())
    summary["statistical_design"] = {"independent_pairs": blocks, "required_pairs": 6,
        "position_counts": positions, "balanced_matrix_complete": balanced,
        "one_pair_is_triage_only": True, "consecutive_updates_are_independent_trials": False}
    if summary["valid"]:
        summary["independent_dense_reuse_attribution"] = workspace
        summary["independent_pinned_pool_layout_attribution"] = arena
        summary["independent_selection_attribution"] = selection
        summary["total_combined_improvement_attribution"] = profile == COMBINED_PROFILE
        ratio_names = set().union(*(row["ratios"] for row in summary["blocks"]))
        summary["ratios"] = {name: paired._statistics([row["ratios"][name] for row in summary["blocks"] if name in row["ratios"]])
                             for name in sorted(ratio_names)}
        delta_names = set().union(*(row["deltas_right_minus_left"] for row in summary["blocks"]))
        summary["deltas_right_minus_left"] = {name: paired._statistics([row["deltas_right_minus_left"][name]
            for row in summary["blocks"] if name in row["deltas_right_minus_left"]]) for name in sorted(delta_names)}
        summary["target_CUDA_evidence_complete"] = (balanced and all(
            str(row.get("device", "")).startswith("cuda") and row.get("physical_target_executed") is True
            and not row.get("diagnostic_overrides") and row.get("model_factory") is None
            and row.get("model", {}).get("parameters") == 32317999
            and row.get("input", {}).get("mode") == "first_dataset_batch_repeated" for row in results))
    else:
        for row in summary["blocks"]:
            row["ratios"] = {}
            row["deltas_right_minus_left"] = {}
    if profile == COMBINED_PROFILE:
        summary["milestone"] = _combined_milestone(summary, plan)
    return summary


def _combined_milestone(summary, plan):
    import math
    import statistics

    result = {**deepcopy(plan["milestone_estimator"]), "assessed": False,
        "memory_threshold_observed": False, "timing_upper_bound_within_one_percent": False,
        "preferred_milestone_estimate_met": False, "performance_policy_promoted": False,
        "scope": "descriptive_same_candidate_R0_to_declared_demand_point_not_global_optimum_or_numerical_qualification"}
    if not summary["valid"]:
        return {**result, "reason": "invalid_paired_cohort"}
    blocks = summary["blocks"]
    reductions = [row["ratios"].get("total_allocated_reduction_percent") for row in blocks]
    if any(type(value) not in (int, float) for value in reductions):
        return {**result, "reason": "whole_run_peak_missing"}
    observed = statistics.mean(reductions)
    result.update(observed_mean_whole_run_allocated_peak_reduction_percent=observed,
                  memory_threshold_observed=observed >= 10)
    if len(blocks) < 6 or not summary["statistical_design"]["balanced_matrix_complete"]:
        return {**result, "reason": "at_least_six_balanced_independent_pairs_required"}
    logs = [math.log(row["ratios"]["runtime_ratio_BEST_over_R0"]) for row in blocks]
    mean, spread = statistics.mean(logs), statistics.stdev(logs)
    upper = math.exp(mean + result["critical_value"] * spread / math.sqrt(len(logs)))
    result.update(assessed=True, independent_pairs=len(logs), mean_paired_log_runtime_ratio=mean,
        sample_stddev_paired_log_runtime_ratio=spread, geometric_mean_runtime_ratio_BEST_over_R0=math.exp(mean),
        one_sided_95_percent_runtime_ratio_upper_bound=upper, timing_upper_bound_within_one_percent=upper <= 1.01,
        target_CUDA_evidence_complete=summary["target_CUDA_evidence_complete"])
    result["preferred_milestone_estimate_met"] = (result["memory_threshold_observed"]
        and result["timing_upper_bound_within_one_percent"] and summary["target_CUDA_evidence_complete"])
    if not result["preferred_milestone_estimate_met"]:
        result["reason"] = "threshold_or_uncertainty_or_target_evidence_not_satisfied"
    return result


def _run_workspace(args, plan):
    from examples.vae import benchmark_vae as benchmark, benchmark_execution as paired, benchmark_packed_offload as offload
    from benchmarks.compiler.installed_runtime import child_environment

    args = deepcopy(args)
    args.trials = plan["trials_requested"]
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    output = args.output.resolve()
    directory = output.parent / (output.stem + "-artifacts")
    if output.exists() or directory.exists():
        raise FileExistsError("Choose a fresh workspace experiment output; historical receipts are immutable")
    directory.mkdir(parents=True)
    report = {**plan, "status": "running", "results": [], "diagnostics": [],
        "configuration_preflight_only": getattr(args, "configuration_preflight_only", False),
        "configuration_preflight": {"valid": False, "workers": [], "invalid_reasons": [],
                                    "scope": "typed_configuration_only_no_model_or_CUDA_execution"},
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    rows = []
    benchmark.write_json(output, report)
    try:
        manifest_data = args.experiment_manifest.read_bytes()
        if _sha(manifest_data) != plan["manifest_sha256"]:
            raise RuntimeError("Workspace manifest changed after planning")
        manifest_copy = directory / "experiment-manifest.json"
        manifest_copy.write_bytes(manifest_data)
        report["experiment_manifest_artifact"] = paired._reference(manifest_copy, output.parent)
        report["helper_source_artifacts"] = []
        for name, digest in plan["source_sha256"].items():
            data = (ROOT / name).read_bytes()
            if _sha(data) != digest:
                raise RuntimeError("Workspace helper source changed after planning: " + name)
            path = directory / "helper-sources" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            report["helper_source_artifacts"].append(paired._reference(path, output.parent))
        environment = child_environment(ROOT, identity=plan["installed"])
        environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
        environment["TORCHDYNAMO_REPRO_LEVEL"] = "0"

        def run_worker(block, arm, configuration=False):
            prefix = ("configuration-" if configuration else f"block-{block}-") + arm
            path, log, command = offload._pinned_command(args, plan, directory, prefix, arm, configuration=configuration)
            from examples.vae import native_execution

            expected = native_execution.execution_plan(native_execution.parser().parse_args(command[3:]))
            if expected["settings"] != plan["workers"][arm]["settings"]:
                raise RuntimeError("Pinned workspace worker differs from its planned configuration")
            pin = json.loads(Path(command[command.index("--arm-manifest") + 1]).read_text())
            print(f"Workspace {'configuration' if configuration else f'pair {block}/{args.trials}'}, arm {arm}", flush=True)
            code, legacy_capacity = None, None
            if plan["experiment_profile"] == ARENA_PROFILE and arm == "H0" and not configuration:
                from examples.vae.execution_memory import host_capacity_admission

                legacy_capacity = host_capacity_admission(2 * plan["manifest"]["policies"][arm]["host_pinned_bytes"],
                    headroom_bytes=plan["host_headroom_bytes"])
                legacy_capacity["scope"] = "conservative_legacy_rounding_envelope_not_a_v1_allocator_cap"
            with log.open("x") as stream:
                if legacy_capacity is not None and not legacy_capacity["admitted"]:
                    limited = {**expected, "status": "failed", "resource_limited": True, "worker_launched": False,
                        "receipt_producer": "paired_supervisor_resource_preflight", "execution_completed": False,
                        "timing_eligible": False, "legacy_host_capacity_preflight": legacy_capacity,
                        "error": "Host resource preflight did not admit the conservative legacy rounding envelope",
                        "failure_context": {"stage": "setup", "phase": "legacy_host_capacity_preflight"}}
                    path.write_text(json.dumps(limited, indent=2, allow_nan=False) + "\n")
                    stream.write(limited["error"] + "\n")
                else:
                    try:
                        code = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                            stderr=subprocess.STDOUT, check=False).returncode
                    except OSError as error:
                        stream.write(str(error) + "\n")
            row = offload._read_worker(path, exit_code=code, block=block, arm=arm)
            row["_expected_arm_pin"] = pin
            if not configuration:
                row.pop("_workspace_history", None)
                if plan["manifest"]["policies"][arm]["enabled"]:
                    try:
                        row["_workspace_history"] = _workspace_read_history(row, path)
                        history_path = directory / (prefix + ".history.json")
                        history_path.write_text(json.dumps(row["_workspace_history"], indent=2, allow_nan=False) + "\n")
                    except (OSError, TypeError, ValueError) as error:
                        row["_workspace_history_error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
            try:
                reasons = (offload._configuration_errors(row, plan, arm) if configuration
                           else _workspace_worker_errors(row, plan, arm))
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                reasons = ["malformed_worker_receipt:" + type(error).__name__ + ":" + str(error)]
            if row.get("origin") != plan["origin"]:
                reasons.append("workspace_origin_changed")
            if row.get("host_capacity_policy") != plan["workers"][arm]["host_capacity_policy"]:
                reasons.append("pinned_host_headroom_policy_mismatch")
            primary = offload._primary_failure(row, reasons)
            if primary and row.get("resource_limited") is True and row.get("worker_launched") is False:
                primary["kind"] = "resource_preflight"
            offload._print_worker_failure(block, arm, primary, configuration=configuration)
            receipt = {"block": block, "arm": arm, "status": row.get("status"), "worker_exit_code": code,
                "error": row.get("error"), "primary_failure": primary, "validation_errors": reasons,
                "log": paired._reference(log, output.parent)}
            if legacy_capacity is not None:
                receipt["legacy_host_capacity_preflight"] = legacy_capacity
                receipt["worker_launched"] = bool(legacy_capacity["admitted"])
            if path.is_file():
                receipt["result"] = paired._reference(path, output.parent)
            if not configuration:
                receipt.update(measurement=row.get("measurement"), memory=row.get("memory"))
                if plan["experiment_profile"] == ARENA_PROFILE and arm == "H0":
                    receipt["legacy_pinned_physical_check"] = _legacy_pinned_check(row, plan["manifest"]["policies"][arm])
                if row.get("_workspace_history"):
                    receipt["canonical_history"] = paired._reference(history_path, output.parent)
            if configuration:
                report["configuration_preflight"]["workers"].append(receipt)
            else:
                report["results"].append(receipt)
            benchmark.write_json(output, report)
            return reasons

        for arm in plan["active_arms"]:
            reasons = run_worker(0, arm, configuration=True)
            report["configuration_preflight"]["invalid_reasons"] += [arm + ":" + reason for reason in reasons]
        preflight = report["configuration_preflight"]
        preflight["valid"] = not preflight["invalid_reasons"]
        if not preflight["valid"]:
            report.update(status="failed", execution_completed=False, timing_eligible=False)
            report["summary"] = {"valid": False, "invalid_reasons": ["configuration_preflight_failed"], "arms": {},
                "ratios": {}, "target_CUDA_evidence_complete": False}
        elif report["configuration_preflight_only"]:
            report.update(status="completed", scope="configuration_preflight_only", execution_completed=False, timing_eligible=False)
            report["summary"] = {"valid": True, "invalid_reasons": [], "arms": {}, "ratios": {}, "target_CUDA_evidence_complete": False}
        else:
            for block, order in enumerate(plan["order"], 1):
                for arm in order:
                    run_worker(block, arm)
            report["summary"] = aggregate_workspace(rows, plan)
            report["status"] = "completed" if report["summary"]["valid"] else "failed"
        for arm in plan["active_arms"]:
            if verify_arm(plan["arms"][arm], args.experiment_manifest.resolve()) != plan["installed"]:
                raise RuntimeError("Pinned workspace installation changed")
        if _sha(args.experiment_manifest.read_bytes()) != plan["manifest_sha256"] or _memory_source_identity() != plan["source_sha256"]:
            raise RuntimeError("Workspace manifest or helper source changed")
    except Exception as error:
        report.update(benchmark.failure_error_details(error), status="failed")
        try:
            report["summary"] = aggregate_workspace(rows, plan)
        except Exception as nested:
            report["summary"] = {"invalid_reasons": [], "aggregation_error": benchmark.failure_error_details(nested)}
        report["summary"].update(valid=False, ratios={}, independent_dense_reuse_attribution=False, target_CUDA_evidence_complete=False)
        report["summary"]["invalid_reasons"].append("supervisor_failed")
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        benchmark.write_json(output, report)
    print(json.dumps(offload._terminal_report(report, output), indent=2))
    return 0 if report["status"] == "completed" else 1


def run_experiment(args):
    from examples.vae import benchmark_execution as paired, benchmark_vae as benchmark

    plan = experiment_plan(args)
    if plan.get("experiment_profile") in PAIRED_MEMORY_PROFILES:
        return _run_workspace(args, plan)
    args = deepcopy(args)
    args.trials = args.trials if args.trials is not None else 3
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    output = args.output.resolve()
    directory = output.parent / (output.stem + "-artifacts")
    if output.exists() or directory.exists():
        raise FileExistsError("Choose a fresh experiment output; historical evidence is immutable")
    directory.mkdir(parents=True)
    manifest_copy = directory / "experiment-manifest.json"
    manifest_data = args.experiment_manifest.read_bytes()
    if _sha(manifest_data) != plan["manifest_sha256"]:
        raise RuntimeError("Experiment manifest changed after planning")
    manifest_copy.write_bytes(manifest_data)
    for name, digest in plan["supervisor_sources"].items():
        data = (ROOT / name).read_bytes()
        if _sha(data) != digest:
            raise RuntimeError("Experiment supervisor source changed after planning")
        target = directory / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    report = {**plan, "status": "running", "results": [], "started_at_utc": datetime.now(timezone.utc).isoformat()}
    report["experiment_manifest_artifact"] = paired._reference(manifest_copy, output.parent)
    benchmark.write_json(output, report)
    results = []
    try:
        for block, order in enumerate(plan["order"], 1):
            for arm in order:
                spec = plan["arms"][arm]
                identity = plan["installed_identities"][arm]
                arm_dir = directory / f"block-{block}-{arm}"
                arm_dir.mkdir()
                options = arm_dir / "native-options.json"
                options.write_text(json.dumps(spec.get("native_options", {}), sort_keys=True) + "\n")
                pin = arm_dir / "arm-manifest.json"
                pin.write_text(json.dumps({"schema_version": 1, "arm": arm,
                    "expected_installed": spec["expected_installed"], "wheel_sha256": spec["wheel"]["sha256"],
                    "native_options_sha256": _sha(options.read_bytes()) if arm != "A" else None}, indent=2) + "\n")
                worker_output, log = arm_dir / "result.json", arm_dir / "worker.log"
                command = paired.worker_command(args, "native", worker_output)
                command.extend(("--arm-manifest", str(pin)))
                if arm != "A":
                    command.extend(("--native-options-json", str(options)))
                from examples.vae import native_execution

                worker_args = native_execution.parser().parse_args(command[3:])
                expected = native_execution.execution_plan(worker_args)
                environment = dict(os.environ)
                environment["PYTHONPATH"] = os.pathsep.join((identity["install_root"], str(ROOT / "benchmarks"), str(ROOT)))
                environment["DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY"] = json.dumps(identity, sort_keys=True)
                environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
                environment["TORCHDYNAMO_REPRO_LEVEL"] = "0"
                print(f"Native compiler experiment block {block}/{args.trials}, arm {arm}", flush=True)
                code = None
                with log.open("x") as stream:
                    try:
                        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                                   stderr=subprocess.STDOUT, check=False)
                        code = completed.returncode
                    except OSError as error:
                        stream.write(str(error) + "\n")
                try:
                    row = json.loads(worker_output.read_text())
                    if not isinstance(row, dict):
                        raise ValueError("worker result is not a JSON object")
                except (OSError, ValueError) as error:
                    row = {"status": "failed", "error": str(error)}
                # Keep the raw receipt unchanged, but do not let malformed
                # nested JSON abort the remaining independent arm processes.
                try:
                    paired._worker_errors(row)
                    compiler = row.get("compiler_report_summary", {})
                    if not isinstance(compiler, dict) or any(
                            not isinstance(compiler.get(key, ()), (list, tuple))
                            or any(not isinstance(item, dict) for item in compiler.get(key, ()))
                            for key in ("fusion_candidates", "effective_compiler_options")):
                        raise ValueError("invalid compiler summary structure")
                except Exception as error:
                    row = {"status": "failed", "error": f"Malformed worker result: {type(error).__name__}: {error}"}
                row.update(arm=arm, block=block, worker_exit_code=code,
                           expected_settings=expected["settings"])
                if row.get("origin", {}).get("sha256") != plan["origin"]["sha256"]:
                    row["status"] = "failed"
                    row.setdefault("error", "Original receipt changed after experiment planning")
                results.append(row)
                receipt = {"block": block, "arm": arm, "status": row.get("status"), "worker_exit_code": code,
                    "log": paired._reference(log, output.parent), "error": row.get("error"),
                    "measurement": row.get("measurement"), "memory": row.get("memory")}
                if worker_output.exists():
                    receipt["result"] = paired._reference(worker_output, output.parent)
                report["results"].append(receipt)
                benchmark.write_json(output, report)
        for arm in "ABC":
            if verify_arm(plan["arms"][arm], args.experiment_manifest.resolve()) != plan["installed_identities"][arm]:
                raise RuntimeError("Pinned experiment installation changed")
        if _sha(args.experiment_manifest.read_bytes()) != plan["manifest_sha256"]:
            raise RuntimeError("Experiment manifest changed")
        if any(_sha((ROOT / path).read_bytes()) != digest for path, digest in plan["supervisor_sources"].items()):
            raise RuntimeError("Experiment supervisor source changed")
        report["summary"] = aggregate_blocks(results, plan["installed_identities"], args.trials,
            start_free_tolerance_bytes=getattr(args, "start_free_tolerance_mib", 256) * 2**20)
        report["status"] = "completed" if report["summary"]["valid"] else "failed"
    except Exception as error:
        report.update(benchmark.failure_error_details(error), status="failed")
        try:
            report["summary"] = aggregate_blocks(results, plan["installed_identities"], args.trials,
                start_free_tolerance_bytes=getattr(args, "start_free_tolerance_mib", 256) * 2**20)
        except Exception as aggregation_error:
            report["summary"] = {"invalid_reasons": [],
                "aggregation_error": benchmark.failure_error_details(aggregation_error)}
        report["summary"].update(valid=False, ratios={}, target_CUDA_evidence_complete=False)
        report["summary"]["invalid_reasons"].append("supervisor_failed")
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        benchmark.write_json(output, report)
    print(json.dumps({"status": report["status"], "summary": report["summary"], "output": str(output)}, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(run_experiment(parser().parse_args()))
