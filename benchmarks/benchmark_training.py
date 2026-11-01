"""Compare a plain PyTorch loop with the functional Trainer on fixed CPU data.

Run with the project's PyTorch environment, for example:
    python benchmarks/benchmark_training.py --profile --output work/benchmark.json
    python benchmarks/benchmark_training.py --checks off --profile --output work/checks-off.json
    python benchmarks/benchmark_training.py --source work/perf-baseline-src-... --profile

Setup and data generation are excluded from timings. Both paths perform the
same scalar-loss SGD updates and report the mean batch loss. Trainer additionally
provides input validation, optional finite checks, metrics, and lifecycle handling.
The --checks switch affects Trainer diagnostics; the plain loop has none.
Profiler allocation totals count repeated allocations; they are NOT peak memory.
"""

import argparse

from dataclasses import dataclass
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time


@dataclass(frozen=True)
class Workload:
    name: str
    input_features: int
    hidden_features: int
    classes: int
    batch_size: int


WORKLOADS = (
    Workload("small_mlp", input_features=16, hidden_features=32, classes=8, batch_size=32),
    Workload("moderate_mlp", input_features=256, hidden_features=512, classes=64, batch_size=128),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="src directory containing deepopt; overrides PYTHONPATH")
    parser.add_argument("--output", type=Path, help="also save the JSON result here")
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--checks", choices=("on", "off"), default="on",
                        help="enable or disable Trainer diagnostics (default: on)")
    parser.add_argument("--profile", action="store_true", help="record CPU operators and allocation totals")
    args = parser.parse_args()
    if args.batches < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("batches/repeats must be positive and warmups nonnegative")
    if args.source is not None:
        source = args.source.resolve()
        if not (source / "deepopt").is_dir():
            parser.error("--source must contain a deepopt package")
        sys.path.insert(0, str(source))
    elif importlib.util.find_spec("deepopt") is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import torch
    from torch import nn
    from torch.nn import functional as F
    from deepopt.engine import Trainer
    try:
        from deepopt.logger import log_output
    except ModuleNotFoundError as exc:
        if exc.name != "deepopt.logger":
            raise
        # --source also benchmarks snapshots predating the logging module.
        from builtins import print as log_output

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    def make_model(workload):
        return nn.Sequential(nn.Linear(workload.input_features, workload.hidden_features), nn.ReLU(),
                             nn.Linear(workload.hidden_features, workload.hidden_features), nn.ReLU(),
                             nn.Linear(workload.hidden_features, workload.classes))

    def profile_run(run):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    profile_memory=True, record_shapes=False) as profile:
            run()
        selected = {"aten::isfinite", "aten::_local_scalar_dense", "aten::detach", "detach", "aten::_to_copy"}
        counts = {name: 0 for name in sorted(selected)}
        for event in profile.key_averages():
            if event.key in counts:
                counts[event.key] = event.count
        return {
            "operator_counts": counts,
            "summed_positive_self_cpu_allocation_bytes": sum(
                max(event.self_cpu_memory_usage, 0) for event in profile.events()),
            "allocation_note": "Cumulative positive operator allocations, not peak or retained memory.",
        }

    results = []
    for workload in WORKLOADS:
        torch.manual_seed(1234)
        initial_state = make_model(workload).state_dict()
        generator = torch.Generator().manual_seed(4321)
        batches = [(torch.randn(workload.batch_size, workload.input_features, generator=generator),
                    torch.randint(workload.classes, (workload.batch_size,), generator=generator))
                   for _ in range(args.batches)]

        def setup(kind):
            model = make_model(workload)
            model.load_state_dict(initial_state)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            if kind == "trainer":
                trainer = Trainer(model, optimizer, loss_fn=F.cross_entropy, checks=args.checks == "on")

                def run():
                    return trainer.train_epoch(batches)["loss"]
            else:
                def run():
                    model.train()
                    total = torch.zeros((), dtype=torch.float64)
                    for inputs, targets in batches:
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(inputs), targets)
                        loss.backward()
                        optimizer.step()
                        total += loss.detach().double()
                    optimizer.zero_grad(set_to_none=True)
                    return (total / len(batches)).item()
            return model, run

        for _ in range(args.warmups):
            for kind in ("plain_pytorch", "trainer"):
                _, run = setup(kind)
                run()
        timings = {"plain_pytorch": [], "trainer": []}
        for repeat in range(args.repeats):
            # Alternate the order to avoid always giving one path the warmer CPU.
            order = ("plain_pytorch", "trainer") if repeat % 2 == 0 else ("trainer", "plain_pytorch")
            for kind in order:
                _, run = setup(kind)
                started = time.perf_counter()
                run()
                timings[kind].append(time.perf_counter() - started)

        plain, run = setup("plain_pytorch")
        plain_loss = run()
        trainer_model, run = setup("trainer")
        trainer_loss = run()
        comparisons = list(zip(plain.parameters(), trainer_model.parameters()))
        matches = all(torch.allclose(left, right, rtol=1e-6, atol=1e-7) for left, right in comparisons)
        loss_matches = (math.isfinite(plain_loss) and math.isfinite(trainer_loss)
                        and math.isclose(plain_loss, trainer_loss, rel_tol=1e-6, abs_tol=1e-7))
        row = {
            "workload": vars(workload),
            "batches": args.batches,
            "optimizer": {"name": "SGD", "learning_rate": 0.01},
            "timings_seconds": {kind: {"median": statistics.median(values), "runs": values}
                                for kind, values in timings.items()},
            "trainer_to_pytorch_median_ratio": statistics.median(timings["trainer"]) / statistics.median(timings["plain_pytorch"]),
            "correctness": {
                "parameters_match": matches,
                "loss_matches": loss_matches,
                "parameter_tolerance": {"rtol": 1e-6, "atol": 1e-7},
                "loss_tolerance": {"rtol": 1e-6, "atol": 1e-7},
                "max_parameter_abs_difference": max((left - right).abs().max().item() for left, right in comparisons),
                "plain_loss": plain_loss,
                "trainer_loss": trainer_loss,
                "loss_abs_difference": abs(plain_loss - trainer_loss),
            },
        }
        if args.profile:
            row["profiles"] = {}
            for kind in ("plain_pytorch", "trainer"):
                _, run = setup(kind)
                row["profiles"][kind] = profile_run(run)
        results.append(row)

    report = {
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(), "device": "cpu", "threads": 1,
                        "deepopt_source": str(Path(importlib.import_module("deepopt").__file__).resolve()),
                        "cuda_available": torch.cuda.is_available()},
        "settings": {"warmups": args.warmups, "repeats": args.repeats, "dtype": "float32",
                     "checks": args.checks == "on",
                     "phase": "training epoch", "data": "pre-generated in-memory batches",
                     "model_seed": 1234, "data_seed": 4321,
                     "timed_setup": False, "profiled_runs_in_timings": False},
        "workloads": results,
        "limitations": "CPU-only timings; GPU speed and peak CPU/GPU memory are not measured.",
    }
    serialized = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    log_output(serialized)
    if not all(row["correctness"]["parameters_match"] and row["correctness"]["loss_matches"] for row in results):
        raise SystemExit("Parameter or loss equivalence failed; do not compare performance as equivalent work.")


if __name__ == "__main__":
    main()
