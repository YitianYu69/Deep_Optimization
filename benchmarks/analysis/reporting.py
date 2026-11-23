"""Matched comparisons, factorial interaction estimates and portable CSV/Markdown."""

from collections import defaultdict
from copy import deepcopy
import csv
import json
import math
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def median_ms(row):
    return row["timing"]["wall_ms"]["median"]


def eager_graph_comparisons(results):
    """Pair only settings that differ in execution, retaining timing exclusions."""
    groups = defaultdict(dict)
    for row in results:
        task = row["task"]
        if task["case"] == "codec" or row.get("kind") == "codec_diagnostic":
            continue
        settings = deepcopy(task.get("settings", {}))
        execution = settings.get("runtime", {}).pop("execution", "eager")
        if execution not in ("eager", "graph"):
            continue
        key = json.dumps([task["workload"], task["case"], settings], sort_keys=True)
        groups[key][execution] = row
    comparisons = []
    for key, modes in groups.items():
        if not {"eager", "graph"} <= modes.keys():
            continue
        eager, graph = modes["eager"], modes["graph"]
        task = graph["task"]
        result = {"workload": task["workload"]["name"], "case": task["case"],
                  "eager_variant_id": eager["task"]["variant_id"], "graph_variant_id": task["variant_id"],
                  "settings_without_execution": json.loads(key)[2]}
        if eager["status"] != "ok" or graph["status"] != "ok":
            comparisons.append({**result, "status": "incomplete", "eager_status": eager["status"],
                                "graph_status": graph["status"]})
            continue
        checks = {}
        for name, field in (("same_initial_state", "initial_state_sha256"),
                            ("same_rng_baseline", "rng_baseline_sha256"),
                            ("same_optimizer_baseline", "shared_optimizer_baseline_sha256"),
                            ("same_measurement_policy", "measurement_policy")):
            checks[name] = eager.get(field) == graph.get(field)
        checks["same_installed_build"] = (eager.get("environment", {}).get("deepopt_source_sha256")
                                            == graph.get("environment", {}).get("deepopt_source_sha256"))
        checks["exclusive_timing"] = all(r.get("scheduling", {}).get("isolation", "exclusive") == "exclusive"
            and r.get("scheduling", {}).get("requested_isolation", "exclusive") == "exclusive" for r in (eager, graph))
        checks["control_drift_passed"] = all(r.get("control_drift", {}).get("passed", True) for r in (eager, graph))
        checks["repeat_variability_passed"] = all(r["timing"]["wall_ms"].get("relative_stdev", 0) <= .03 for r in (eager, graph))
        checks["finite"] = all(r.get("finite_after_timing", True) for r in (eager, graph))
        checks["compression_coverage_verified"] = all(r.get("compression_coverage", {}).get("status") != "unverified" for r in (eager, graph))
        evidence = graph.get("execution", {}).get("cuda_graph")
        if evidence is not None:
            checks["graph_replay_observed"] = evidence.get("captures", 0) > 0 and evidence.get("replays", 0) > 0
        # Report an autotuner disagreement separately: the observed ratio then
        # includes the launch choice and is not attributable to replay alone.
        launches = [r.get("execution", {}).get("convolution_launches") for r in (eager, graph)]
        same_launches = launches[0] == launches[1] if all(v is not None for v in launches) else None
        eager_ms, graph_ms = median_ms(eager), median_ms(graph)
        accepted = all(checks.values())
        comparisons.append({**result, **checks, "status": "ok" if accepted else "unqualified",
            "eager_ms": eager_ms, "graph_ms": graph_ms,
            "raw_graph_speedup": eager_ms / graph_ms, "performance_graph_speedup": eager_ms / graph_ms if accepted else None,
            "latency_reduction_percent": 100 * (1 - graph_ms / eager_ms),
            "same_convolution_launches": same_launches,
            "eager_peak_allocated_bytes": (eager.get("memory") or {}).get("peak_allocated_bytes"),
            "graph_reserved_bytes": (graph.get("graph_memory") or {}).get("reserved_with_graph_bytes")})
    return comparisons


def summarize(results, tolerances, measured_epochs=None, *, numerical_comparisons=None):
    from .measurement import errors
    import torch

    groups = defaultdict(dict)
    for row in results:
        if row.get("kind") != "codec_diagnostic" and row["task"]["case"] != "codec":
            groups[(row["task"]["variant_id"], row["task"]["workload"]["name"])][row["task"]["case"]] = row
    comparisons, interactions = [], []
    for (variant, name), cases in groups.items():
        native = cases.get("torch_bf16")
        snapshots = {}
        for case, row in cases.items():
            if (row["status"] == "ok" and row.get("snapshot") and row.get("comparison_input") is None
                    and row.get("measurement_policy", {}).get("numerical_snapshot_mode") != "memory"):
                snapshots[case] = torch.load(row["snapshot"], map_location="cpu", weights_only=True)
        spec = next(iter(cases.values()))["task"]["workload"]
        policy = tolerances.get(spec["op"], tolerances.get("default", {"rtol": .05, "atol": .01}))
        for case, row in cases.items():
            if row["status"] != "ok" or case == "torch_bf16" or native is None or native["status"] != "ok":
                continue
            same = row["initial_state_sha256"] == native["initial_state_sha256"]
            same_optimizer = row.get("shared_optimizer_baseline_sha256") == native.get("shared_optimizer_baseline_sha256")
            same_rng = row.get("rng_baseline_sha256") == native.get("rng_baseline_sha256")
            same_measurement = row.get("measurement_policy") == native.get("measurement_policy")
            exclusive = all(r.get("scheduling",{}).get("isolation","exclusive")=="exclusive"
                            and r.get("scheduling",{}).get("requested_isolation","exclusive")=="exclusive" for r in (row,native))
            drift = all(r.get("control_drift",{}).get("passed",True) for r in (row,native))
            same_build = row.get("environment", {}).get("deepopt_source_sha256") == native.get("environment", {}).get("deepopt_source_sha256")
            memory_required = any(r.get("comparison_input") is not None
                                  or r.get("measurement_policy", {}).get("numerical_snapshot_mode") == "memory"
                                  for r in (row, native))
            checked = memory_required or case in snapshots and "torch_bf16" in snapshots
            numerical = (numerical_comparisons.compare_rows(row, native, **policy,
                             gradient_policy=tolerances.get("gradients"), update_policy=tolerances.get("updates"))
                         if memory_required and numerical_comparisons is not None else
                         {"passed": False, "finite": False, "reason": "missing_live_numerical_comparison"}
                         if memory_required else
                         errors(snapshots[case], snapshots["torch_bf16"], **policy,
                                gradient_policy=tolerances.get("gradients"), update_policy=tolerances.get("updates"))
                         if checked else {"status": "not_requested", "passed": None, "finite": True})
            numerical["full_tensor_finite"] = (row.get("numerical_snapshot", {}).get("full_tensor_finite", True)
                                                and native.get("numerical_snapshot", {}).get("full_tensor_finite", True))
            numerical["same_output_and_gradient_shapes"] = ((row.get("numerical_snapshot", {}).get("original_shapes", {})
                                                              == native.get("numerical_snapshot", {}).get("original_shapes", {}))
                                                             if checked else None)
            numerical["finite_after_timing"] = row.get("finite_after_timing", True) and native.get("finite_after_timing", True)
            if checked:
                numerical["passed"] &= numerical["finite_after_timing"]
                numerical["passed"] &= numerical["full_tensor_finite"] and numerical["same_output_and_gradient_shapes"]
                numerical["passed"] &= same_rng and same and same_optimizer and same_build
            numerical["finite"] &= numerical["finite_after_timing"]
            numerical["finite"] &= numerical["full_tensor_finite"]
            if memory_required:
                numerical["scope"] = "descriptive_quantization_quality_vs_torch_bf16"
            row["correctness_vs_torch_bf16"] = numerical
            uncompressed = {"compression": "do_bf16", "fp8_compression": "fp8"}.get(case)
            if memory_required and uncompressed in cases:
                row["correctness_vs_uncompressed"] = (numerical_comparisons.compare_rows(row, cases[uncompressed], **policy,
                    gradient_policy=tolerances.get("gradients"), update_policy=tolerances.get("updates"))
                    if numerical_comparisons is not None else {"passed": False, "finite": False,
                        "reason": "missing_live_numerical_comparison"})
                row["correctness_vs_uncompressed"]["scope"] = "descriptive_quantization_quality_vs_uncompressed"
            elif case in snapshots and uncompressed in snapshots:
                row["correctness_vs_uncompressed"] = errors(snapshots[case], snapshots[uncompressed], **policy,
                    gradient_policy=tolerances.get("gradients"), update_policy=tolerances.get("updates"))
            raw = median_ms(native) / median_ms(row)
            coverage = row.get("compression_coverage", {}).get("status") != "unverified"
            stable = all(r["timing"]["wall_ms"].get("relative_stdev", 0) <= .03 for r in (row, native))
            timing_valid = same and same_rng and same_build and same_optimizer and same_measurement and exclusive and drift and stable and numerical["finite"] and coverage
            comparisons.append({"variant_id": variant, "workload": name, "case": case,
                                "same_initial_state": same, "accuracy_passed": numerical["passed"],
                                "same_rng_baseline": same_rng,
                                "same_installed_build": same_build,
                                "same_optimizer_baseline": same_optimizer, "same_measurement_policy": same_measurement,
                                "exclusive_timing": exclusive,
                                "control_drift_passed":drift,
                                "repeat_variability_passed": stable,
                                "compression_coverage_verified": coverage,
                                "finite": numerical["finite"], "raw_wall_speedup_vs_torch": raw,
                                "performance_wall_speedup_vs_torch": raw if timing_valid else None,
                                "qualified_wall_speedup_vs_torch": raw if timing_valid and numerical["passed"] else None,
                                "extra_wall_ms": median_ms(row) - median_ms(native)})
        required = ("do_bf16", "compression", "fp8", "fp8_compression")
        missing = [key for key in required if cases.get(key, {}).get("status") != "ok"]
        if missing:
            interactions.append({"variant_id": variant, "workload": name, "status": "incomplete", "missing": missing})
            continue
        b, c, d, e = (median_ms(cases[key]) for key in required)
        same = len({cases[key]["initial_state_sha256"] for key in required}) == 1
        same_optimizer = len({cases[key].get("shared_optimizer_baseline_sha256") for key in required}) == 1
        same_measurement = (len({json.dumps(cases[key].get("measurement_policy"), sort_keys=True) for key in required}) == 1
                            and all(cases[key].get("scheduling",{}).get("isolation","exclusive")=="exclusive"
                                    and cases[key].get("scheduling",{}).get("requested_isolation","exclusive")=="exclusive" for key in required))
        same_build = len({cases[key].get("environment", {}).get("deepopt_source_sha256") for key in required}) == 1
        same_rng = len({cases[key].get("rng_baseline_sha256") for key in required}) == 1
        stable = all(cases[key]["timing"]["wall_ms"].get("relative_stdev", 0) <= .03 for key in required)
        drift = all(cases[key].get("control_drift", {}).get("passed", True) for key in required)
        finite = all(cases[key].get("finite_after_timing", True)
                     and cases[key].get("numerical_snapshot", {}).get("full_tensor_finite", True)
                     and all(value is None or bool(torch.isfinite(value).all()) for value in snapshots.get(key, {}).values())
                     for key in required)
        interaction = e - d - c + b
        # A descriptive uncertainty estimate for independent repeat means; medians are the reported estimator.
        se = math.sqrt(sum(cases[key]["timing"]["wall_ms"]["stdev"] ** 2 /
                           len(cases[key]["timing"]["wall_ms"]["samples"]) for key in required))
        rows = {key: cases[key]["timing"]["cuda_stream_ms"] for key in required}
        cuda_interaction = None
        if all(rows.values()):
            cuda_interaction = rows["fp8_compression"]["median"] - rows["fp8"]["median"] - rows["compression"]["median"] + rows["do_bf16"]["median"]
        calls, steps = spec.get("calls_per_step"), spec.get("steps_per_epoch")
        phases = {}
        for phase in ("forward_and_loss", "backward"):
            for metric in ("cuda_stream_ms", "host_submission_ms"):
                samples = [(cases[key].get("phases") or {}).get(phase, {}).get(metric) for key in required]
                if all(samples):
                    x, y, z, w = (sample["median"] for sample in samples)
                    phases[f"{phase}.{metric}"] = w - z - y + x
        accuracy_values = [cases[key].get("correctness_vs_torch_bf16", {}).get("passed") for key in required]
        accuracy = all(accuracy_values) if all(value is not None for value in accuracy_values) else None
        coverage = all(cases[key].get("compression_coverage", {}).get("status") != "unverified"
                       for key in ("compression", "fp8_compression"))
        timing_valid = same and same_build and same_optimizer and same_rng and same_measurement and stable and drift and finite and coverage
        interactions.append({"variant_id": variant, "workload": name,
                             "status": "ok" if timing_valid else "unqualified",
                             "same_optimizer_baseline": same_optimizer, "same_measurement_policy": same_measurement,
                             "same_rng_baseline": same_rng, "repeat_variability_passed": stable,
                             "control_drift_passed": drift,
                             "compression_coverage_verified": coverage,
                             "same_initial_state": same, "finite": finite,
                             "same_installed_build": same_build,
                             "interaction_wall_ms": interaction, "interaction_cuda_stream_ms": cuda_interaction,
                             "performance_interaction_wall_ms": interaction if timing_valid else None,
                             "independent_repeat_mean_standard_error_ms": se,
                             "noise_warning": not timing_valid or abs(interaction) <= 3 * se or any(
                                 len(cases[key]["timing"]["wall_ms"]["samples"]) < 3 for key in required),
                             "accuracy_passed": accuracy, "phase_interactions_ms": phases,
                             "compression_cost_bf16_ms": c - b, "compression_cost_fp8_ms": e - d,
                             "combined_additive_prediction_ms": d + c - b, "combined_observed_ms": e,
                             "estimated_interaction_ms_per_model_step": interaction * calls if calls is not None else None,
                             "estimated_interaction_seconds_per_epoch": interaction * calls * steps / 1000
                             if calls is not None and steps is not None else None,
                             "qualification": "A timing interaction is not proof of a kernel cause. Validate with real-model ablations. "
                                              "Call-count weighting assumes isolated costs transfer and ignores overlap/fusion."})
    interactions.sort(key=lambda r: -r.get("interaction_wall_ms", -math.inf))
    observed = None
    if measured_epochs:
        b, c, d, e = (measured_epochs[key] for key in ("bf16", "compression", "fp8", "combined"))
        observed = {"epoch_seconds": measured_epochs, "interaction_seconds": e - d - c + b,
                    "additive_prediction_seconds": d + c - b,
                    "interaction_fraction_of_total_overhead": (e - d - c + b) / (e - b) if e != b else None,
                    "source": "user-supplied observations, not measurements from this suite"}
    weighted = sorted((row for row in interactions if row["status"] == "ok" and row.get("estimated_interaction_ms_per_model_step") is not None),
                      key=lambda row: -row["estimated_interaction_ms_per_model_step"])
    checked_comparisons = [c for c in comparisons if c["accuracy_passed"] is not None]
    qualification = {"comparisons_checked": len(checked_comparisons),
                     "performance_comparisons": len(comparisons),
                     "performance_timing_accepted": sum(c["performance_wall_speedup_vs_torch"] is not None for c in comparisons),
                     "numerical_not_requested": len(comparisons) - len(checked_comparisons),
                     "bf16_numerical_accepted": sum(c["accuracy_passed"] for c in checked_comparisons),
                     "bf16_numerical_rejected": sum(not c["accuracy_passed"] for c in checked_comparisons),
                     "codec_oracle": "not_measured_by_this_matrix",
                     "training_quality": "not_qualified_by_single_update_comparisons",
                     "status": "completed_with_numerical_rejections" if any(not c["accuracy_passed"] for c in checked_comparisons)
                               else "no_numerical_rejections" if checked_comparisons else "performance_only" if comparisons else "not_compared"}
    return {"comparisons": comparisons, "eager_vs_graph": eager_graph_comparisons(results),
            "interactions": interactions, "ranked_by_model_contribution": weighted,
            "qualification": qualification,
            "observed_epoch_context": observed,
            "counts": {status: sum(row["status"] == status for row in results) for status in ("ok", "skipped", "error")}}


def write_reports(directory, report):
    directory = Path(directory)
    write_json(directory / "summary.json", report)
    graph_columns = ("workload", "case", "eager_variant_id", "graph_variant_id", "status", "eager_ms", "graph_ms",
                     "raw_graph_speedup", "performance_graph_speedup", "latency_reduction_percent", "same_convolution_launches",
                     "eager_peak_allocated_bytes", "graph_reserved_bytes")
    with (directory / "eager-vs-graph.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=graph_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report.get("eager_vs_graph", []))
    quality_rows = []
    for row in report["results"]:
        quality = row.get("quality_diagnostics") or {}
        task = row["task"]
        base = {"case": task["case"], "workload": task["workload"]["name"],
                "variant_id": task["variant_id"], "seed": quality.get("seed"),
                "validation_status": quality.get("validation_status")}
        quality_rows.extend({**base, "step": item["step"], "metric": "training_loss", "value": item["training_loss"]}
                            for item in quality.get("curve", []))
        quality_rows.extend({**base, "step": item["step"], "metric": name, "value": value}
                            for item in quality.get("validation", []) for name, value in item["metrics"].items())
    with (directory / "quality-curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "workload", "variant_id", "seed", "validation_status", "step", "metric", "value"))
        writer.writeheader()
        writer.writerows(quality_rows)
    columns = ("variant_id", "workload", "op", "case", "status", "wall_ms", "cuda_stream_ms",
               "host_submission_ms", "host_end_event_record_ms", "host_event_synchronize_ms",
               "host_event_elapsed_query_ms", "wall_minus_cuda_stream_ms", "producer_pack_hits", "producer_pack_misses",
               "peak_allocated_bytes", "observed_saved_storage_bytes", "accuracy_passed", "reason")
    with (directory / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["results"]:
            task = row["task"]
            timing = row.get("timing", {}) if task["case"] != "codec" else {}
            writer.writerow({"variant_id": task["variant_id"], "workload": task["workload"]["name"],
                             "op": task["workload"]["op"], "case": task["case"], "status": row["status"],
                             "wall_ms": (timing.get("wall_ms") or {}).get("median"),
                             "cuda_stream_ms": (timing.get("cuda_stream_ms") or {}).get("median"),
                             **{name: (timing.get(name) or {}).get("median") for name in
                                ("host_submission_ms", "host_end_event_record_ms", "host_event_synchronize_ms",
                                 "host_event_elapsed_query_ms", "wall_minus_cuda_stream_ms")},
                             "producer_pack_hits": row.get("execution", {}).get("producer_pack_reuse", {}).get("hits"),
                             "producer_pack_misses": row.get("execution", {}).get("producer_pack_reuse", {}).get("misses"),
                             "peak_allocated_bytes": (row.get("memory") or {}).get("peak_allocated_bytes"),
                             "observed_saved_storage_bytes": (row.get("memory") or {}).get("observed_saved_unique_storage_bytes"),
                             "accuracy_passed": row.get("correctness_vs_torch_bf16", {}).get("passed"),
                             "reason": row.get("reason", "")})
    lines = ["# DO / PyTorch BF16 analysis", "", f"Results: {report['counts']}", "",
             f"Numerical qualification: {report.get('qualification', 'unavailable')}", "",
             "Positive interaction means FP8 and compression together cost more than the sum of their separate overheads.", "",
             "| Workload | Variant | Interaction ms/update | Compression on BF16 ms | Compression on FP8 ms | Status |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in report["interactions"]:
        if "interaction_wall_ms" in row:
            lines.append(f"| {row['workload']} | {row['variant_id']} | {row['interaction_wall_ms']:.4f} | "
                         f"{row['compression_cost_bf16_ms']:.4f} | {row['compression_cost_fp8_ms']:.4f} | "
                         f"{row['status']}{' / noisy' if row['noise_warning'] else ''}"
                         f"{' / accuracy rejected' if row['accuracy_passed'] is False else ''} |")
        else:
            lines.append(f"| {row['workload']} | {row['variant_id']} | — | — | — | Missing: {', '.join(row['missing'])} |")
    lines.extend(("", "## Matched eager versus CUDA Graph", "",
                  "Ratios include the complete timed update. Different convolution launches are disclosed in eager-vs-graph.csv; "
                  "those ratios include autotuner choices. Precision comparisons are optional and do not gate performance timing.", "",
                  "| Workload | Case | Eager ms | Graph ms | Speedup | Status |",
                  "| --- | --- | ---: | ---: | ---: | --- |"))
    for row in report.get("eager_vs_graph", []):
        if "eager_ms" in row:
            lines.append(f"| {row['workload']} | {row['case']} | {row['eager_ms']:.4f} | {row['graph_ms']:.4f} | "
                         f"{row['raw_graph_speedup']:.3f}× | {row['status']} |")
        else:
            lines.append(f"| {row['workload']} | {row['case']} | — | — | — | {row['status']} |")
    lines.extend(("", "## Accuracy and exclusions", "",
                  "Raw timing ratios remain visible. Performance speedups require matched, finite, stable exclusive timing; "
                  "accuracy-qualified speedups additionally require passing numerical comparisons. "
                  "Inspect summary.json for per-output/input-gradient/parameter-gradient/update errors, exclusions and traces.", ""))
    for row in report["results"]:
        if row["status"] != "ok":
            task = row["task"]
            reason = row.get("reason", "").replace("\n", " ")
            lines.append(f"- {task['workload']['name']} / {task['case']} / {task['variant_id']}: {row['status']} — {reason}")
    lines.extend(("", "CUDA stream time includes submission gaps. Kernel activity sums are not step latency. "
                  "CPU runs do not establish CUDA throughput. All-family ablations control for ablation-hook overhead.", ""))
    (directory / "report.md").write_text("\n".join(lines))
