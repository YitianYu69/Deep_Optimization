"""Export the current benchmark sources without selecting a DeepOpt source tree."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from analysis.benchmark_do_ops import DEFAULT_TOLERANCES
from analysis.catalog import workload_list
from analysis.matrix import DEFAULT_SETTINGS, EXTENDED_GRID, merge, variants


def main(argv=None):
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--output", type=Path, default=ROOT / "work/full-comparison-benchmarks",
                           help="Destination for separate scripts, configs, guide and a sibling ZIP")
    OUTPUT = arguments.parse_args(argv).output.resolve()
    benchmark_root = (ROOT / "benchmarks").resolve()
    if OUTPUT == benchmark_root or benchmark_root in OUTPUT.parents:
        arguments.error("Choose an export directory outside benchmarks so canonical sources cannot be overwritten")
    EXPORTED_FILES = set()
    OUTPUT.mkdir(parents=True, exist_ok=True)


    def write_json(name, data):
        (OUTPUT / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        EXPORTED_FILES.add(name)


    header = '''#!/usr/bin/env python3
"""Portable full DeepOpt / Torch BF16 benchmark.

This file contains all benchmark helper sources below. It materializes only
those helpers into a temporary directory while running, including isolated
worker processes. It imports the installed DeepOpt package; no library source
tree is added to Python's search path. See RUN_BENCHMARKS.txt for comparisons.
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


_BENCHMARK_SOURCES = {
'''
    sources = []
    for source in sorted((ROOT / "benchmarks/analysis").glob("*.py")):
        # Keep the embedded code readable, including its original docstrings.
        literal = "'''" + source.read_text(encoding="utf-8").replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "'''"
        sources.append(f"    {source.name!r}: {literal},\n")

    footer = '''}


def main(argv=None):
    with TemporaryDirectory(prefix="deepopt-benchmark-") as temporary:
        package = Path(temporary) / "analysis"
        package.mkdir()
        for name, source in _BENCHMARK_SOURCES.items():
            (package / name).write_text(source, encoding="utf-8")
        sys.path.insert(0, temporary)
        try:
            from analysis.benchmark_do_ops import main as run_benchmark
            return run_benchmark(argv)
        finally:
            sys.path.remove(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
'''
    (OUTPUT / "benchmark_do_ops.py").write_text(header + "".join(sources) + footer, encoding="utf-8")
    EXPORTED_FILES.add("benchmark_do_ops.py")

    for relative in ("benchmarks/activation_compression/benchmark_activation_unpack.py",
                     "benchmarks/fp8/benchmark_fp8_linear_compare.py"):
        shutil.copyfile(ROOT / relative, OUTPUT / Path(relative).name)
        EXPORTED_FILES.add(Path(relative).name)

    # Keep independently editable originals as well as the single-file launcher.
    source_directory = OUTPUT / "analysis"
    source_directory.mkdir(exist_ok=True)
    for source in sorted((ROOT / "benchmarks/analysis").glob("*.py")):
        shutil.copyfile(source, source_directory / source.name)
        EXPORTED_FILES.add("analysis/" + source.name)

    for config in sorted((ROOT / "benchmarks/analysis/configs").glob("*.json")):
        data = json.loads(config.read_text())
        for name, policy in DEFAULT_TOLERANCES.items():
            data.setdefault("tolerances", {}).setdefault(name, deepcopy(policy))
        write_json(config.name, data)

    all_operators = {"settings": deepcopy(DEFAULT_SETTINGS), "workloads": workload_list(),
                     "tolerances": deepcopy(DEFAULT_TOLERANCES)}
    write_json("all_operators.json", all_operators)

    all_settings = deepcopy(all_operators)
    all_settings["grid"] = deepcopy(EXTENDED_GRID)
    all_settings["grid"]["compression.group_size"] = [32, 128, 512, 1024, 2048, 32768]
    write_json("all_settings.json", all_settings)

    vit = json.loads((OUTPUT / "h100_vit_recovery.json").read_text())
    vit_ablation = deepcopy(vit)
    vit_ablation["grid"] = {"runtime.compression_families": [None, [], ["linear"], ["attention"],
        ["normalization"], ["activation"], ["convolution"], ["pooling"],
        ["linear", "attention", "normalization", "activation", "convolution", "pooling"]]}
    write_json("h100_vit_ablation.json", vit_ablation)

    # Cross only the primary regression factors; the broad all-settings grid is OAT.
    vit_interactions = deepcopy(vit)
    vit_interactions["grid"] = {"runtime.execution": ["eager", "graph"],
        "fp8.backward_backend": ["inherit", "native_fast", "native_precise"],
        "compression.group_size": [512, 1024], "compression.bits": [2, 4],
        "compression.norm_xhat_recenter": [False, True]}
    write_json("h100_vit_interactions.json", vit_interactions)

    settings_variants = len(list(variants(all_settings["settings"], all_settings["grid"], "one-at-a-time")))
    vit_interaction_tasks = 5 * len(list(variants(
        merge(DEFAULT_SETTINGS, vit_interactions["settings"]),
        vit_interactions["grid"], "cartesian")))
    operators = sum(not item["op"].endswith("codec") for item in all_operators["workloads"])
    codecs = len(all_operators["workloads"]) - operators
    counts = {"operator_workloads": operators, "codec_workloads": codecs,
              "all_operator_tasks_with_gradient_probes": (operators * 5 + codecs) * 5,
              "settings_axes": len(all_settings["grid"]), "settings_variants": settings_variants,
              "all_settings_tasks": settings_variants * (operators * 5 + codecs),
              "vit_cases": 5, "vit_family_ablation_tasks": 45, "vit_crossed_interaction_tasks": vit_interaction_tasks}

    runner = r'''#!/usr/bin/env bash
# Run on one otherwise idle NVIDIA GPU with the UPDATED DeepOpt build installed.
set -uo pipefail
benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
benchmark_python="${BENCHMARK_PYTHON:-python}"
benchmark_output="${1:-full-comparison-results}"
benchmark_profile_steps="${BENCHMARK_PROFILE_STEPS:-3}"
benchmark_status=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

run_benchmark() {
  "$@" || benchmark_status=1
}

# Every supported operator, full backward and individual requested gradients.
run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
  --config "$benchmark_dir/all_operators.json" --gradient-modes all dx dw db none \
  --profile-steps "$benchmark_profile_steps" --observed-epoch-seconds 21 27 22 47 \
  --output "$benchmark_output/operators"

# Every declared setting axis for all operator/pipeline/codec workloads.
run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
  --config "$benchmark_dir/all_settings.json" --sweep one-at-a-time \
  --max-experiments 100000 --profile-steps "$benchmark_profile_steps" \
  --output "$benchmark_output/settings"

# Fully cross the finite input/output/scale/block precision-boundary grid.
run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
  --config "$benchmark_dir/precision_boundaries.json" --sweep cartesian \
  --profile-steps "$benchmark_profile_steps" --output "$benchmark_output/precision-boundaries"

# Five complete ViT cases, then family ablations, then the crossed hot factors.
for benchmark_vit_config in h100_vit_recovery h100_vit_ablation h100_vit_interactions; do
  benchmark_sweep=one-at-a-time
  if [[ "$benchmark_vit_config" == h100_vit_interactions ]]; then benchmark_sweep=cartesian; fi
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
    --config "$benchmark_dir/$benchmark_vit_config.json" --sweep "$benchmark_sweep" \
    --require-fp8-plan --deterministic --fill-uninitialized-memory --capture-workloads \
    --warmup 10 --iterations 30 --repeats 5 --profile-steps "$benchmark_profile_steps" \
    --output "$benchmark_output/$benchmark_vit_config"
done

# Shared-operand GEMMs plus complete BF16 / current / delayed FP8 Linear updates.
# Add torchao to --implementations separately if it is already installed.
for benchmark_backward_backend in inherit native_fast native_precise; do
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_fp8_linear_compare.py" \
    --implementations bf16 deepopt-current deepopt-delayed \
    --backward-backend "$benchmark_backward_backend" --prepared-gemms \
    --profile-steps "$benchmark_profile_steps" \
    --output "$benchmark_output/linear-$benchmark_backward_backend"
done

# Same packed payload: original grouped, tiled, decode/copy and production auto.
for benchmark_source_dtype in bf16 e4m3fn e5m2; do
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_activation_unpack.py" \
    --source-dtype "$benchmark_source_dtype" --dtypes bf16 e4m3fn e5m2 \
    --bits 1 2 4 8 --group-sizes 128 512 1024 2048 --execution both \
    --output "$benchmark_output/unpack-$benchmark_source_dtype.json"
done
exit "$benchmark_status"
'''
    (OUTPUT / "run_full_comparisons.sh").write_text(runner, encoding="utf-8")
    EXPORTED_FILES.add("run_full_comparisons.sh")

    guide = f'''FULL DEEPOPT / TORCH BF16 COMPARISONS

Run this directory on the H100 with the updated DeepOpt build installed.
The benchmark files import the installed library. They do not install packages,
select a repository source tree, or download pretrained models or datasets.
The unpack controls need the optimized build that supports the implementation
argument; they cannot run against the unmodified older codec.

FILES
benchmark_do_ops.py: self-contained full five-case operator/model suite.
benchmark_fp8_linear_compare.py: BF16/current FP8/delayed FP8 Linear updates,
  plus shared-operand Triton/native-fast/native-precise GEMM probes.
benchmark_activation_unpack.py: original grouped/tiled/copy/auto reconstruction.
all_operators.json: {operators} operator/pipeline workloads and {codecs} codec controls.
all_settings.json: all workloads over {counts['settings_axes']} setting axes / {settings_variants} OAT variants.
h100_vit_recovery.json: ViT base patch16/224, batch512, ten classes, AdamW/EMA.
h100_vit_ablation.json: all/none/each compression family, including hook controls.
h100_vit_interactions.json: crossed eager/graph, backward backend, group,
  bit width and recentering factors ({vit_interaction_tasks} requested tasks).
precision_boundaries.json: crossed input/output types, scales, FP8 formats/blocks.
interaction.json: original six regression workloads with new accuracy policies.
settings_sweep.json / model_ablation.json: smaller diagnostic configs.
analysis/: editable module sources matching the embedded runner. To use edits,
  run python analysis/benchmark_do_ops.py with the same config/CLI arguments.
manifest.json: file/source hashes, settings and workload counts.

The five cases are torch_bf16, do_bf16, compression, fp8, fp8_compression.
Codecs use separately labeled storage/roundtrip controls, not training cases.

FULL RUN
  bash run_full_comparisons.sh /path/to/results

This is a large run. It continues subsequent benchmark sections if one fails,
preserves the failure reports, and exits nonzero if any section failed.
The all-operator gradient probes request {counts['all_operator_tasks_with_gradient_probes']} tasks; the
all-settings run requests {counts['all_settings_tasks']} tasks. Unsupported combinations remain
explicit skipped/error rows. They are not silently counted as measured coverage.
Runs use separate workers, sequentially on one GPU; no concurrent measurements.
Set BENCHMARK_PYTHON=/path/to/python to select the environment.
Set BENCHMARK_PROFILE_STEPS=0 to omit the separate profiler passes if needed.
Default full-run profile steps are 3; traces from this large sweep need disk space.

INDIVIDUAL RUNS (commands below assume the current directory contains these files)

1. Validate the grid without Torch/CUDA or executing kernels:
  python benchmark_do_ops.py --config all_settings.json --max-experiments 100000 \\
    --dry-run --output plan-settings

2. Every operator, all five cases, full and requested-gradient probes:
  python benchmark_do_ops.py --config all_operators.json \\
    --gradient-modes all dx dw db none --profile-steps 3 --output results/operators

3. All workloads and all declared setting axes, changing one factor at a time:
  python benchmark_do_ops.py --config all_settings.json --sweep one-at-a-time \\
    --max-experiments 100000 --profile-steps 3 --output results/settings

4. Fully crossed finite precision-boundary comparisons:
  python benchmark_do_ops.py --config precision_boundaries.json \\
    --sweep cartesian --profile-steps 3 --output results/boundaries

5. Full ViT five-case comparison:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark_do_ops.py \\
    --config h100_vit_recovery.json --require-fp8-plan --deterministic \\
    --fill-uninitialized-memory --capture-workloads --warmup 10 \\
    --iterations 30 --repeats 5 --profile-steps 3 --output results/vit

6. Compression-family attribution with the same ViT factory:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark_do_ops.py \\
    --config h100_vit_ablation.json --require-fp8-plan --deterministic \\
    --fill-uninitialized-memory --profile-steps 3 --output results/vit-families

7. Cross the principal ViT regression factors:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark_do_ops.py \\
    --config h100_vit_interactions.json --sweep cartesian --require-fp8-plan \\
    --deterministic --fill-uninitialized-memory --profile-steps 3 \\
    --output results/vit-interactions

8. Complete Linear and prepared shared-operand GEMM controls:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark_fp8_linear_compare.py \\
    --implementations bf16 deepopt-current deepopt-delayed --prepared-gemms \\
    --profile-steps 3 --output results/linear
  Optional: append torchao to --implementations if already installed.
  Repeat --backward-backend native_fast and native_precise to change the full
  training executor. Prepared probes compare those GEMMs in the same invocation.

9. Reconstruction controls at the dominant original H100 shapes:
  python benchmark_activation_unpack.py --bits 1 2 4 8 \\
    --group-sizes 128 512 1024 2048 --dtypes bf16 e4m3fn e5m2 \\
    --execution both --output results/unpack.json
  Repeat --source-dtype bf16 / e4m3fn / e5m2 and optional --block 32x32.

RESULTS AND INTERPRETATION
The full operator/model suite writes report.md, results.csv, summary.json,
plan.json, per-case result.json, worker logs, numerical snapshots and traces.
It retains synchronized wall and CUDA stream samples, forward/loss/backward
phases, optimizer/update timing, memory/saved storage, output/input-gradient/
parameter-gradient/update accuracy, CPU operators, GPU kernel names/launches,
and the FP8/compression timing interaction:
  combined - fp8 - compression + do_bf16
Separate profiler passes do not instrument the throughput timing loops. For graph
cases the profiler diagnoses the eager path and is labeled accordingly; graph
throughput measures replay. Stream event times include host submission gaps.
Raw timings remain visible when an accuracy policy rejects a case. Qualified
speedups additionally require matching initial state/build and passing policies.
Independent gradient/update relative-L2 and cosine gates are enabled in configs.
Use --strict-accuracy to make numerical rejection fail the operator/model CLI.
Use --max-numerical-elements 0 for full rather than sampled numerical diagnostics.
Gradient modes dx/dw/db request separate complete executions; their times cannot
be added to reconstruct the backward time. Parameterless probes are skipped.

OAT tests all listed values individually; it does not test every interaction or
every possible tensor size/integer option. The explicit Cartesian configs test
their declared cross-products. To add a suspected interaction, narrow a JSON
grid and use --sweep cartesian; do not cross the entire broad settings grid.
Operator shapes/kwargs are editable. Captured native module workloads appear
in recorded-workloads.json; replay them with --config for model-specific shapes.
This replay is representative and cannot reconstruct every functional call,
arbitrary stride or saved tensor. Use the real-model result as the authority.

DEPENDENCIES AND BEFORE/AFTER
Use the original compatible CUDA PyTorch/Triton environment plus updated DeepOpt.
Full ViT needs timm and its dependencies; hf_conv1d additionally needs transformers.
Missing optional dependencies and unsupported operations remain visible as skips.
No automatic dependency installation occurs. --help and --dry-run need only Python.
The full ViT factory uses fresh untrained weights and resident data. It includes
AdamW and EMA, but does not reproduce the original trainer's metrics, gradient
clones or process-group orchestration. Rerun that original diagnostic for an
exact before/after comparison against the supplied trainer traces.
For old/new library comparisons, keep this benchmark code/config/device/seed/
environment policy fixed, run in separate installed-library environments, and
write different output directories. The old build's known failures are evidence;
do not disable accuracy/planner checks or replace failed rows with new timings.
CPU smoke tests validate portability/logic only; these exports have not been
measured on the H100 here and do not establish a GPU speedup.
'''
    (OUTPUT / "RUN_BENCHMARKS.txt").write_text(guide, encoding="utf-8")
    EXPORTED_FILES.add("RUN_BENCHMARKS.txt")

    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "cases": [
        "torch_bf16", "do_bf16", "compression", "fp8", "fp8_compression"], "coverage": counts,
        "files": {}, "canonical_analysis_sources": {}}
    for source in sorted((ROOT / "benchmarks/analysis").glob("*.py")):
        manifest["canonical_analysis_sources"][source.name] = sha256(source.read_bytes()).hexdigest()
    for name in sorted(EXPORTED_FILES):
        manifest["files"][name] = sha256((OUTPUT / name).read_bytes()).hexdigest()
    write_json("manifest.json", manifest)

    archive = OUTPUT.parent / (OUTPUT.name + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(EXPORTED_FILES):
            bundle.write(OUTPUT / name, OUTPUT.name + "/" + name)
    print(json.dumps({"directory": str(OUTPUT), "archive": str(archive), "coverage": counts}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
