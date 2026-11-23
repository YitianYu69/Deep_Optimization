#!/usr/bin/env python3
"""Run matched allocator experiments in fresh CUDA processes, without importing Torch here."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def allocator_environment(expandable, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    primary, alias = (env.get(k) for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"))
    if primary and alias and primary != alias:
        raise ValueError("Conflicting allocator environment aliases; supply one consistent startup configuration")
    options = primary or alias or ""
    backend = re.search(r"(?:^|,)\s*backend\s*:\s*([^,]+)", options)
    if backend and backend.group(1).strip() != "native":
        raise ValueError("Expandable-segment comparison requires backend:native")
    options = re.sub(r"(?:^|,)\s*expandable_segments\s*:\s*[^,]+", "", options).strip(",")
    if not backend:
        options = "backend:native" + ("," + options if options else "")
    options += ",expandable_segments:" + ("True" if expandable else "False")
    # Both aliases agree, including versions preferring the canonical name.
    env["PYTORCH_ALLOC_CONF"] = env["PYTORCH_CUDA_ALLOC_CONF"] = options
    return env


def main(argv=None):
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--variable-config", type=Path, help="Optional eager factory with changing resident batch sizes")
    p.add_argument("--runner", type=Path, default=root / "benchmark_do_ops.py")
    p.add_argument("--output", type=Path, default=Path("allocator-comparison"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--profile-steps", type=int, default=3)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-numerical-elements", type=int, default=65536)
    p.add_argument("--dry-run", action="store_true")
    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=False)
    args = p.parse_args(argv)
    if min(args.profile_steps,args.max_numerical_elements) < 0 or min(args.iterations, args.repeats) < 1:
        p.error("iterations/repeats must be positive and profile steps nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    try:
        for label, source in [("fixed", args.config), ("variable", args.variable_config)]:
            if source is None:
                continue
            config = json.loads(source.read_text())
            runtime = config.setdefault("settings", {}).setdefault("runtime", {})
            runtime["timing_fill_uninitialized_memory"] = False
            # A fill sweep in the source must not override this experiment's
            # fixed performance policy after cartesian/OAT expansion.
            config.get("grid", {}).pop("runtime.timing_fill_uninitialized_memory", None)
            if label == "variable":
                runtime["execution"] = "eager"
            for enabled in (False, True):
                env = allocator_environment(enabled)
                name = f"{label}-expandable-{'on' if enabled else 'off'}"
                directory = args.output / name
                directory.mkdir(parents=True, exist_ok=True)
                config_path = directory / "input-config.json"
                config_path.write_text(json.dumps(config, indent=2) + "\n")
                command = [args.python, str(args.runner.resolve()), "--config", str(config_path.resolve()),
                           "--sweep", "cartesian",
                           "--output", str((directory / "results").resolve()), "--deterministic",
                           "--fill-uninitialized-memory", "--warmup", "10", "--iterations", str(args.iterations),
                           "--repeats", str(args.repeats), "--profile-steps", str(args.profile_steps),
                           "--max-numerical-elements",str(args.max_numerical_elements),
                           "--profile-replay-steps", "1" if args.profile_steps else "0"]
                if args.resume:
                    command.append("--resume")
                job = {"name": name, "command": command,
                       "allocator_environment": {k: env[k] for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
                       "performance_fill": False, "correctness_fill": True}
                jobs.append(job)
                if not args.dry_run:
                    with (directory / "runner.log").open("w") as log:
                        job["exit_code"] = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    except (ValueError, OSError) as error:
        p.error(str(error))
    (args.output / "allocator-experiment.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(json.dumps(jobs, indent=2))
    return int(any(job.get("exit_code", 0) for job in jobs))


if __name__ == "__main__":
    raise SystemExit(main())
