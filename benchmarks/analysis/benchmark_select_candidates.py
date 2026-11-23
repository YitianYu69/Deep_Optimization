"""Export stable candidate settings for matched profiling; never promote defaults."""

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.matrix import fingerprint
from analysis.reporting import write_json


def exclusion(row):
    if row.get("status") != "ok":
        return "execution_not_ok"
    wall = row.get("timing", {}).get("wall_ms") or {}
    if len(wall.get("samples", [])) < 3 or wall.get("relative_stdev", float("inf")) > .03:
        return "repeat_variability"
    if not row.get("finite_after_timing", False):
        return "nonfinite_or_unchecked"
    if not row.get("control_drift", {}).get("passed", True):
        return "control_drift"
    scheduling = row.get("scheduling", {})
    if any(scheduling.get(key, "exclusive") != "exclusive" for key in ("isolation", "requested_isolation")):
        return "shared_timing"
    if row.get("compression_coverage", {}).get("status") == "unverified":
        return "compression_unverified"
    return None


def select_candidates(report):
    groups, excluded = defaultdict(list), []
    for row in report["results"]:
        task = row["task"]
        if task["case"] not in ("compression", "fp8", "fp8_compression"):
            continue
        settings = task["settings"]
        if settings["runtime"]["execution"] not in ("eager", "graph"):
            continue
        reason = exclusion(row)
        if reason:
            excluded.append({"task_id": task["id"], "reason": reason})
            continue
        arithmetic = {"fp8": {k: v for k, v in settings["fp8"].items() if k != "num_stages"},
                      "compression": settings["compression"],
                      "runtime": {k: settings["runtime"].get(k) for k in
                          ("input_dtype", "output_dtype", "input_scaling", "output_scaling", "sdpa_backend",
                           "attention_backend", "linear_dw_backend", "linear_forward_backend",
                           "compression_bits_by_family", "compression_families", "compression_exclude_modules",
                           "optimizer", "optimizer_options", "optimizer_warmup_steps", "lr", "accumulation_steps")}}
        key = (fingerprint(task["workload"]), task["case"], settings["runtime"]["execution"], fingerprint(arithmetic))
        groups[key].append(row)
    selections = []
    for rows in groups.values():
        identities = set()
        for row in rows:
            identities.add(json.dumps([row.get("initial_state_sha256"), row.get("rng_baseline_sha256"),
                row.get("shared_optimizer_baseline_sha256"), row.get("measurement_policy"),
                row.get("environment", {}).get("deepopt_source_sha256")], sort_keys=True))
        shared_optimizer = rows[0].get("shared_optimizer_baseline_sha256")
        warmed_optimizer = rows[0]["task"]["settings"]["runtime"].get("optimizer_warmup_steps", 0)
        if (len(identities) != 1 or not rows[0].get("environment", {}).get("deepopt_source_sha256")
                or any(not rows[0].get(key) for key in ("initial_state_sha256", "rng_baseline_sha256"))
                or "shared_optimizer_baseline_sha256" not in rows[0]
                or (warmed_optimizer and not shared_optimizer)):
            excluded.extend({"task_id": r["task"]["id"], "reason": "incomparable_baselines"} for r in rows)
            continue
        winner = min(rows, key=lambda r: (r["timing"]["wall_ms"]["median"], r["task"]["id"]))
        selections.append(winner)
    return selections, excluded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.summary.read_bytes()
    report = json.loads(raw)
    rows, excluded = select_candidates(report)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_summary_sha256": sha256(raw).hexdigest(), "selections": [], "excluded": excluded,
                "note": "Fastest stable screen candidates, not default promotions or proof of a statistically significant gain. "
                        "Profile and confirm using matched controls and longer exclusive timings."}
    for row in rows:
        task = row["task"]
        config = {"settings": deepcopy(task["settings"]), "workloads": [deepcopy(task["workload"])]}
        if "tolerances" in report:
            config["tolerances"] = deepcopy(report["tolerances"])
        name = "selected-" + fingerprint(config) + ".json"
        write_json(args.output / name, config)
        manifest["selections"].append({"config": name, "task_id": task["id"], "case": task["case"],
            "workload": task["workload"]["name"], "wall_ms": row["timing"]["wall_ms"]["median"],
            "source_sha256": row.get("environment", {}).get("deepopt_source_sha256"),
            "compute_coverage": row.get("execution", {}).get("convolution_coverage"),
            "producer_reuse": [r for r in row.get("execution", {}).get("codec_dispatches", [])
                               if r.get("operation") == "producer_pack_consume"]})
    write_json(args.output / "selection.json", manifest)
    (args.output / "selected-configs.txt").write_text("".join(name + "\n" for name in
        sorted({row["config"] for row in manifest["selections"]})))
    print(json.dumps({"selected": len(rows), "excluded": len(excluded), "output": str(args.output)}))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
