#!/usr/bin/env bash
# Matched speed comparisons; numerical-precision diagnostics are opt-in.
set -uo pipefail
benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
benchmark_python="${BENCHMARK_PYTHON:-python}"
benchmark_output=runtime-speed-results
benchmark_output_seen=0
benchmark_resume=0
benchmark_final=0
for argument in "$@"; do
  case "$argument" in
    --resume) benchmark_resume=1;;
    --final) benchmark_final=1;;
    --*) printf 'Unknown argument: %s\n' "$argument" >&2; exit 2;;
    *) if [[ "$benchmark_output_seen" == 1 ]]; then printf '%s\n' 'Supply one output directory' >&2; exit 2; fi
       benchmark_output="$argument"; benchmark_output_seen=1;;
  esac
done
benchmark_stage="${SPEED_STAGE:-compare}"
case "$benchmark_stage" in compare|profiles|cnn|convolution|vit|attention|references|training|all) ;;
  *) printf '%s\n' 'SPEED_STAGE must be compare/profiles/cnn/convolution/vit/attention/references/training/all' >&2; exit 2;;
esac
benchmark_backend="${SPEED_ATTENTION_BACKEND:-flash_attn_2}"
case "$benchmark_backend" in native|flash_attn_2) ;; *) printf '%s\n' 'SPEED_ATTENTION_BACKEND must be native or flash_attn_2' >&2; exit 2;; esac
benchmark_configs="$benchmark_dir"
if [[ -d "$benchmark_dir/configs" ]]; then benchmark_configs="$benchmark_dir/configs"; fi
benchmark_warmup=5; benchmark_iterations=10; benchmark_repeats=3
benchmark_cpu_workers="${BENCHMARK_CPU_WORKERS:-4}"
if [[ "$benchmark_final" == 1 ]]; then benchmark_warmup=10; benchmark_iterations=30; benchmark_repeats=5; fi
if [[ "$benchmark_final" == 1 && -z "${BENCHMARK_CPU_WORKERS:-}" ]]; then benchmark_cpu_workers=1; fi
benchmark_status=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

benchmark_preflight_env=(env)
if [[ -n "${BENCHMARK_DEVICES:-}" && "$BENCHMARK_DEVICES" != auto ]]; then
  benchmark_preflight_env=(env "CUDA_VISIBLE_DEVICES=${BENCHMARK_DEVICES%%,*}")
fi
"${benchmark_preflight_env[@]}" "$benchmark_python" "$benchmark_dir/benchmark_attention_preflight.py" --runtime-speed \
  --output "$benchmark_output/preflight.json" || exit 1

run_benchmark() {
  if [[ "$benchmark_resume" == 1 ]]; then "$@" --resume || benchmark_status=1
  else "$@" || benchmark_status=1; fi
}
run_matrix() {
  local config="$1" sweep="$2" profile="$3" replay="$4"
  shift 4
  local config_path="$benchmark_configs/$config.json"
  if [[ "$config" == *.json ]]; then config_path="$config"; fi
  local extra=(--scheduler auto)
  if [[ -n "${BENCHMARK_DEVICES:-}" ]]; then extra+=(--devices "$BENCHMARK_DEVICES"); fi
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
    --config "$config_path" --sweep "$sweep" --no-numerical-comparison \
    --warmup "${BENCHMARK_WARMUP:-$benchmark_warmup}" --iterations "${BENCHMARK_ITERATIONS:-$benchmark_iterations}" \
    --repeats "${BENCHMARK_REPEATS:-$benchmark_repeats}" --seed 14 --phase-steps 3 \
    --latency-steps "${BENCHMARK_LATENCY_STEPS:-0}" \
    --profile-steps "$profile" --profile-replay-steps "$replay" \
    --profile-policy selected --trace-cases torch_bf16 compression fp8 fp8_compression \
    --cpu-workers "$benchmark_cpu_workers" --gpu-concurrency auto \
    --timing-isolation "${BENCHMARK_TIMING_ISOLATION:-exclusive}" --deduplicate-effective-settings \
    --output "$benchmark_output/$config" "${extra[@]}" "$@"
}
if [[ "$benchmark_stage" == compare || "$benchmark_stage" == all ]]; then
  benchmark_config=h100_runtime_speed_comparison
  if [[ "$benchmark_backend" == native ]]; then benchmark_config=h100_runtime_speed_native; fi
  run_matrix "$benchmark_config" cartesian 0 0
fi
if [[ "$benchmark_stage" == profiles || "$benchmark_stage" == all ]]; then
  if [[ -n "${SPEED_SELECTED_CONFIGS:-}" ]]; then
    benchmark_selected_count=0
    if [[ ! -f "$SPEED_SELECTED_CONFIGS/selected-configs.txt" ]]; then
      printf '%s\n' 'Run benchmark_select_candidates.py first: selected-configs.txt is missing' >&2
      exit 1
    fi
    while IFS= read -r benchmark_selected_file; do
      if [[ "$benchmark_selected_file" != selected-*.json || "$benchmark_selected_file" == */* ]]; then
        printf '%s\n' 'Invalid selected configuration filename' >&2; exit 1
      fi
      benchmark_selected="$SPEED_SELECTED_CONFIGS/$benchmark_selected_file"
      if [[ ! -f "$benchmark_selected" ]]; then printf '%s\n' 'Selected configuration is missing' >&2; exit 1; fi
      benchmark_selected_count=$((benchmark_selected_count + 1))
      benchmark_selected_name="$(basename "$benchmark_selected" .json)"
      # Each selected setting retains all five feature controls.
      run_matrix "$benchmark_selected" cartesian 1 1 --output "$benchmark_output/profiles-$benchmark_selected_name"
    done < "$SPEED_SELECTED_CONFIGS/selected-configs.txt"
    if [[ "$benchmark_selected_count" == 0 ]]; then
      printf '%s\n' 'SPEED_SELECTED_CONFIGS contains no selected-*.json configurations' >&2
      benchmark_status=1
    fi
  else
    run_matrix h100_runtime_speed_profiles cartesian 1 1
  fi
fi
if [[ "$benchmark_stage" == cnn || "$benchmark_stage" == all ]]; then
  run_matrix h100_runtime_speed_cnn one-at-a-time 0 0
fi
if [[ "$benchmark_stage" == convolution || "$benchmark_stage" == all ]]; then
  run_matrix h100_runtime_speed_convolution cartesian 0 0
fi
if [[ "$benchmark_stage" == vit || "$benchmark_stage" == all ]]; then
  run_matrix h100_runtime_speed_vit one-at-a-time 0 0
fi
if [[ "$benchmark_stage" == attention || "$benchmark_stage" == all ]]; then
  run_matrix h100_runtime_speed_attention_vit one-at-a-time 0 0
  run_matrix h100_runtime_speed_attention_blocks one-at-a-time 0 0
fi
if [[ "$benchmark_stage" == references ]]; then
  run_matrix h100_runtime_speed_compiled cartesian 0 0 --cases torch_bf16
  benchmark_linear_script="$benchmark_dir/benchmark_fp8_linear_compare.py"
  if [[ ! -f "$benchmark_linear_script" ]]; then benchmark_linear_script="$benchmark_dir/../fp8/benchmark_fp8_linear_compare.py"; fi
  benchmark_reference_device="${BENCHMARK_REFERENCE_DEVICE:-cuda:0}"
  benchmark_reference_env=(env)
  if [[ -z "${BENCHMARK_REFERENCE_DEVICE:-}" && -n "${BENCHMARK_DEVICES:-}" && "$BENCHMARK_DEVICES" != auto ]]; then
    benchmark_reference_env=(env "CUDA_VISIBLE_DEVICES=${BENCHMARK_DEVICES%%,*}")
    benchmark_reference_device=cuda:0
  fi
  run_benchmark "${benchmark_reference_env[@]}" "$benchmark_python" "$benchmark_linear_script" --device "$benchmark_reference_device" \
    --implementations bf16 deepopt-current torchao --math-policies amp --seed 14 \
    --no-deterministic --no-fill-uninitialized-memory --prepared-gemms \
    --warmups "${BENCHMARK_WARMUP:-$benchmark_warmup}" --iterations "${BENCHMARK_ITERATIONS:-$benchmark_iterations}" \
    --repeats "${BENCHMARK_REPEATS:-$benchmark_repeats}" --profile-steps 0 --no-trace --no-host-profile \
    --output "$benchmark_output/linear-references"
fi
if [[ "$benchmark_stage" == training ]]; then
  if [[ -z "${RECOVERY_TRAIN_CONFIG:-}" || -z "${RECOVERY_TRAIN_OPTIMIZER:-}" ]]; then
    printf '%s\n' 'Set RECOVERY_TRAIN_CONFIG and RECOVERY_TRAIN_OPTIMIZER; keep your configured LR.' >&2; exit 2
  fi
  for model in vit_base_patch16_224 resnet18 efficientnet_b0; do
    library=timm
    if [[ "$model" == resnet18 ]]; then library=torchvision; fi
    for execution in eager graph; do
      graph_flag=--no-graph
      if [[ "$execution" == graph ]]; then graph_flag=--graph; fi
      args=("$benchmark_python" "$benchmark_dir/benchmark_vit_attention_stability.py"
        --config "$RECOVERY_TRAIN_CONFIG" --optimizer "$RECOVERY_TRAIN_OPTIMIZER" \
        --model-name "$model" --model-library "$library" \
        --dataset-root "${RECOVERY_DATASET_ROOT:-data}" \
        --batch-size 512 --drop-last --epochs "${RECOVERY_TRAIN_EPOCHS:-2}" "$graph_flag" \
        --attention-pack auto --activation-layout auto --convolution-kernel-variant auto \
        --profile-train-steps "${RECOVERY_TRAIN_PROFILE_STEPS:-0}" \
        --output "$benchmark_output/training-$model-$execution")
      if [[ -n "${RECOVERY_TRAIN_INITIALIZER:-}" ]]; then args+=(--initializer "$RECOVERY_TRAIN_INITIALIZER"); fi
      run_benchmark "${args[@]}"
    done
  done
fi
exit "$benchmark_status"
