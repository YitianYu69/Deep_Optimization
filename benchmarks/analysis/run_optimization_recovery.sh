#!/usr/bin/env bash
# Focused candidate screening; final comparisons use larger repeat counts.
set -uo pipefail
benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
benchmark_python="${BENCHMARK_PYTHON:-python}"
benchmark_output="${1:-optimization-recovery-results}"
benchmark_resume=0
if [[ "${2:-}" == --resume ]]; then benchmark_resume=1; fi
benchmark_stage="${RECOVERY_STAGE:-all}"
benchmark_status=0
benchmark_config_dir="$benchmark_dir"
if [[ -d "$benchmark_dir/configs" ]]; then benchmark_config_dir="$benchmark_dir/configs"; fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
case "$benchmark_stage" in all|profiles|convolution|vit|attention|quality|training) ;; *)
  printf '%s\n' 'RECOVERY_STAGE must be all/profiles/convolution/vit/attention/quality/training' >&2; exit 2;; esac

"$benchmark_python" "$benchmark_dir/benchmark_attention_preflight.py" \
  --output "$benchmark_output/preflight.json" || exit 1

run_benchmark() {
  if [[ "$benchmark_resume" == 1 ]]; then "$@" --resume || benchmark_status=1
  else "$@" || benchmark_status=1; fi
}
run_matrix() {
  local config="$1" sweep="$2" profile="$3" replay_profile="$4"
  shift 4
  if [[ -n "${BENCHMARK_DEVICES:-}" ]]; then set -- "$@" --devices "$BENCHMARK_DEVICES"; fi
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
    --config "$benchmark_config_dir/$config.json" --sweep "$sweep" \
    --warmup "${BENCHMARK_WARMUP:-5}" --iterations "${BENCHMARK_ITERATIONS:-10}" \
    --repeats "${BENCHMARK_REPEATS:-3}" --seed 14 --phase-steps 3 \
    --profile-steps "$profile" --profile-replay-steps "$replay_profile" \
    --profile-policy selected --trace-cases torch_bf16 compression fp8 fp8_compression \
    --cpu-workers "${BENCHMARK_CPU_WORKERS:-4}" --gpu-concurrency auto \
    --timing-isolation "${BENCHMARK_TIMING_ISOLATION:-exclusive}" --deduplicate-effective-settings \
    --output "$benchmark_output/$config" "$@"
}
if [[ "$benchmark_stage" == all || "$benchmark_stage" == profiles ]]; then
  run_matrix h100_optimization_hotspots cartesian 1 1 --require-fp8-plan
fi
if [[ "$benchmark_stage" == all || "$benchmark_stage" == convolution ]]; then
  run_matrix h100_convolution_candidates cartesian 0 0 --require-fp8-plan
fi
if [[ "$benchmark_stage" == all || "$benchmark_stage" == vit ]]; then
  run_matrix h100_vit_packing_candidates cartesian 0 0 --require-fp8-plan
fi
if [[ "$benchmark_stage" == all || "$benchmark_stage" == attention ]]; then
  run_matrix h100_attention_block_candidates one-at-a-time 0 0 --cases torch_bf16 do_bf16 compression
fi
if [[ "$benchmark_stage" == quality ]]; then
  run_matrix h100_update_alignment cartesian 0 0 --require-fp8-plan
fi
if [[ "$benchmark_stage" == training ]]; then
  if [[ -z "${RECOVERY_TRAIN_CONFIG:-}" || -z "${RECOVERY_TRAIN_OPTIMIZER:-}" ]]; then
    printf '%s\n' 'Set RECOVERY_TRAIN_CONFIG and RECOVERY_TRAIN_OPTIMIZER; the stable LR must come from that config.' >&2
    exit 2
  fi
  for benchmark_model in vit_base_patch16_224 resnet18 efficientnet_b0; do
    benchmark_library=timm
    if [[ "$benchmark_model" == resnet18 ]]; then benchmark_library=torchvision; fi
    benchmark_train_args=("$benchmark_python" "$benchmark_dir/benchmark_vit_attention_stability.py" \
      --config "$RECOVERY_TRAIN_CONFIG" --optimizer "$RECOVERY_TRAIN_OPTIMIZER" \
      --model-name "$benchmark_model" --model-library "$benchmark_library" \
      --dataset-root "${RECOVERY_DATASET_ROOT:-data}" \
      --batch-size 512 --epochs "${RECOVERY_TRAIN_EPOCHS:-2}" --graph --drop-last \
      --attention-pack auto --profile-train-steps "${RECOVERY_TRAIN_PROFILE_STEPS:-0}" \
      --output "$benchmark_output/training-$benchmark_model")
    if [[ -n "${RECOVERY_TRAIN_INITIALIZER:-}" ]]; then
      benchmark_train_args+=(--initializer "$RECOVERY_TRAIN_INITIALIZER")
    fi
    run_benchmark "${benchmark_train_args[@]}"
  done
fi
exit "$benchmark_status"
