#!/usr/bin/env bash
# Focused replay/layout recovery; each timed stage uses an exclusive GPU lease.
set -uo pipefail
benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
benchmark_python="${BENCHMARK_PYTHON:-python}"
benchmark_output="${1:-attention-recovery-results}"
benchmark_resume=0
if [[ "${2:-}" == --resume ]]; then benchmark_resume=1; fi
benchmark_status=0
benchmark_config_dir="$benchmark_dir"
if [[ -d "$benchmark_dir/configs" ]]; then benchmark_config_dir="$benchmark_dir/configs"; fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Never launch a long matrix after a broken forward/backward/graph smoke test.
"$benchmark_python" "$benchmark_dir/benchmark_attention_preflight.py" \
  --output "$benchmark_output/preflight.json" || exit 1

run_benchmark() {
  if [[ "$benchmark_resume" == 1 ]]; then
    "$@" --resume || benchmark_status=1
  else
    "$@" || benchmark_status=1
  fi
}

for benchmark_pack in separate joint; do
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_attention_alignment.py" \
    --batch-size 512 --source-layout interleaved --attention-pack "$benchmark_pack" \
    --codecs current legacy_rules --graph --profile --iterations 30 --repeats 5 \
    --output "$benchmark_output/alignment-$benchmark_pack"
done

benchmark_pack_script="$benchmark_dir/benchmark_attention_pack.py"
if [[ ! -f "$benchmark_pack_script" ]]; then benchmark_pack_script="$benchmark_dir/../activation_compression/benchmark_attention_pack.py"; fi
# Preserve the three independent former group slots under the 1024 recipe.
for benchmark_group_slot in 0 1 2; do
  benchmark_pack_output="$benchmark_output/packing.json"
  if [[ "$benchmark_group_slot" != 0 ]]; then
    benchmark_pack_output="$benchmark_output/packing-group1024-slot-$benchmark_group_slot.json"
  fi
  run_benchmark "$benchmark_python" "$benchmark_pack_script" --qkv-only \
    --bits 4 8 --group-sizes 1024 --source-layouts bhsd bshd interleaved \
    --indexing wide bounded --num-warps 2 4 8 --iterations 30 --repeats 5 \
    --output "$benchmark_pack_output"
done

for benchmark_config in hopper_replay_models hopper_replay_vit; do
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
    --config "$benchmark_config_dir/$benchmark_config.json" --sweep cartesian \
    --warmup 10 --iterations 30 --repeats 5 --seed 14 --profile-steps 0 --require-fp8-plan \
    --cpu-workers 4 --gpu-concurrency auto --timing-isolation exclusive --deduplicate-effective-settings \
    --output "$benchmark_output/$benchmark_config"
done
exit "$benchmark_status"
