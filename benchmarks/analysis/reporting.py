"""Matched comparisons, factorial interaction estimates and portable CSV/Markdown."""

from collections import defaultdict
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


def summarize(results, tolerances, measured_epochs=None):
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
            if row["status"] == "ok":
                snapshots[case] = torch.load(row["snapshot"], map_location="cpu", weights_only=True)
        spec = next(iter(cases.values()))["task"]["workload"]
        policy = tolerances.get(spec["op"], tolerances.get("default", {"rtol": .05, "atol": .01}))
        for case, row in cases.items():
            if row["status"] != "ok" or case == "torch_bf16" or native is None or native["status"] != "ok":
                continue
            same = row["initial_state_sha256"] == native["initial_state_sha256"]
            same_build = row.get("environment", {}).get("deepopt_source_sha256") == native.get("environment", {}).get("deepopt_source_sha256")
            numerical = errors(snapshots[case], snapshots["torch_bf16"], **policy,
                               gradient_policy=tolerances.get("gradients"), update_policy=tolerances.get("updates"))
            numerical["full_tensor_finite"] = (row.get("numerical_snapshot", {}).get("full_tensor_finite", True)
                                                and native.get("numerical_snapshot", {}).get("full_tensor_finite", True))
            numerical["same_output_and_gradient_shapes"] = (row.get("numerical_snapshot", {}).get("original_shapes", {})
                                                             == native.get("numerical_snapshot", {}).get("original_shapes", {}))
            numerical["finite_after_timing"] = row.get("finite_after_timing", True) and native.get("finite_after_timing", True)
            numerical["passed"] &= numerical["finite_after_timing"]
            numerical["passed"] &= numerical["full_tensor_finite"] and numerical["same_output_and_gradient_shapes"]
            numerical["finite"] &= numerical["finite_after_timing"]
            numerical["finite"] &= numerical["full_tensor_finite"]
            row["correctness_vs_torch_bf16"] = numerical
            uncompressed = {"compression": "do_bf16", "fp8_compression": "fp8"}.get(case)
            if uncompressed in snapshots:
                row["correctness_vs_uncompressed"] = errors(snapshots[case], snapshots[uncompressed], **policy,
                    gradient_policy=tolerances.get("gradients"), update_policy=tolerances.get("updates"))
            raw = median_ms(native) / median_ms(row)
            comparisons.append({"variant_id": variant, "workload": name, "case": case,
                                "same_initial_state": same, "accuracy_passed": numerical["passed"],
                                "same_installed_build": same_build,
                                "finite": numerical["finite"], "raw_wall_speedup_vs_torch": raw,
                                "qualified_wall_speedup_vs_torch": raw if same and same_build and numerical["passed"] else None,
                                "extra_wall_ms": median_ms(row) - median_ms(native)})
        required = ("do_bf16", "compression", "fp8", "fp8_compression")
        missing = [key for key in required if cases.get(key, {}).get("status") != "ok"]
        if missing:
            interactions.append({"variant_id": variant, "workload": name, "status": "incomplete", "missing": missing})
            continue
        b, c, d, e = (median_ms(cases[key]) for key in required)
        same = len({cases[key]["initial_state_sha256"] for key in required}) == 1
        same_build = len({cases[key].get("environment", {}).get("deepopt_source_sha256") for key in required}) == 1
        finite = all(cases[key].get("finite_after_timing", True)
                     and cases[key].get("numerical_snapshot", {}).get("full_tensor_finite", True)
                     and all(value is None or bool(torch.isfinite(value).all()) for value in snapshots[key].values())
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
        accuracy = all(cases[key].get("correctness_vs_torch_bf16", {}).get("passed", False) for key in required)
        interactions.append({"variant_id": variant, "workload": name,
                             "status": "ok" if same and same_build and finite else "unqualified",
                             "same_initial_state": same, "finite": finite,
                             "same_installed_build": same_build,
                             "interaction_wall_ms": interaction, "interaction_cuda_stream_ms": cuda_interaction,
                             "independent_repeat_mean_standard_error_ms": se,
                             "noise_warning": abs(interaction) <= 3 * se or any(
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
    weighted = sorted((row for row in interactions if row.get("estimated_interaction_ms_per_model_step") is not None),
                      key=lambda row: -row["estimated_interaction_ms_per_model_step"])
    return {"comparisons": comparisons, "interactions": interactions, "ranked_by_model_contribution": weighted,
            "observed_epoch_context": observed,
            "counts": {status: sum(row["status"] == status for row in results) for status in ("ok", "skipped", "error")}}


def write_reports(directory, report):
    directory = Path(directory)
    write_json(directory / "summary.json", report)
    columns = ("variant_id", "workload", "op", "case", "status", "wall_ms", "cuda_stream_ms",
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
                             "peak_allocated_bytes": (row.get("memory") or {}).get("peak_allocated_bytes"),
                             "observed_saved_storage_bytes": (row.get("memory") or {}).get("observed_saved_unique_storage_bytes"),
                             "accuracy_passed": row.get("correctness_vs_torch_bf16", {}).get("passed"),
                             "reason": row.get("reason", "")})
    lines = ["# DO / PyTorch BF16 analysis", "", f"Results: {report['counts']}", "",
             "Positive interaction means FP8 and compression together cost more than the sum of their separate overheads.", "",
             "| Workload | Variant | Interaction ms/update | Compression on BF16 ms | Compression on FP8 ms | Status |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in report["interactions"]:
        if "interaction_wall_ms" in row:
            lines.append(f"| {row['workload']} | {row['variant_id']} | {row['interaction_wall_ms']:.4f} | "
                         f"{row['compression_cost_bf16_ms']:.4f} | {row['compression_cost_fp8_ms']:.4f} | "
                         f"{row['status']}{' / noisy' if row['noise_warning'] else ''}"
                         f"{' / accuracy rejected or unavailable' if not row['accuracy_passed'] else ''} |")
        else:
            lines.append(f"| {row['workload']} | {row['variant_id']} | — | — | — | Missing: {', '.join(row['missing'])} |")
    lines.extend(("", "## Accuracy and exclusions", "",
                  "Raw timing ratios remain visible. Qualified speedups require matching initial state and passing numerical tolerances. "
                  "Inspect summary.json for per-output/input-gradient/parameter-gradient/update errors, exclusions and traces.", ""))
    for row in report["results"]:
        if row["status"] != "ok":
            task = row["task"]
            reason = row.get("reason", "").replace("\n", " ")
            lines.append(f"- {task['workload']['name']} / {task['case']} / {task['variant_id']}: {row['status']} — {reason}")
    lines.extend(("", "CUDA stream time includes submission gaps. Kernel activity sums are not step latency. "
                  "CPU runs do not establish CUDA throughput. All-family ablations control for ablation-hook overhead.", ""))
    (directory / "report.md").write_text("\n".join(lines))
