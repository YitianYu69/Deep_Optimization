# Code generation, final schedules and invocation storage

The compiler reports its final execution schedule and its storage lifetimes.

## Final schedules and cleanup

The [scheduler](../../src/deepopt/compiler/scheduler.py) resolves declared fused dX placement constraints after differentiation and fusion, before partitioning and liveness analysis. This is a narrow explicit-order implementation, not an automatic latency/memory optimizer. Real data dependencies, unsafe values and opaque effects are barriers. RNG, normalization updates, FP8 observations, communication and arbitrary external calls do not gain reordering permission.

The [joint-region report](../../src/deepopt/compiler/autodiff/joint.py) records requested backward order, differentiation-selected role order, order immediately after fusion, and final emitted order. If a later coefficient binding is the only impediment, its input binding can move with the fused dX. If a dependency or effect prevents the requested order, the report explains the restriction. Report rows describe static operations, not observed CUDA launches.

[MemoryManager](../../src/deepopt/compiler/runtime/memory.py) rebuilds per-dispatch consumer counts from the final schedule, releases unused results at birth, and emits local reference deletion after last use. Rewriting a schedule after analysis invalidates code generation. Logical lifetime records preserve lifted parameter/buffer ownership and output escape status. Their byte counts must not be summed into a physical peak: alias groups, actual backing storage and opaque workspace remain runtime questions.

This implementation does not reuse scratch storage. A local `del` is not device-completion evidence and does not authorize buffer aliasing, cross-stream reuse or capture-pool reuse.

## Storage and capacity evidence

The invocation-local [storage ledger](../../src/deepopt/memory/ledger.py) uses weak storage references and generations. It separates physical backing allocations from logical views, records strides/offsets/version boundaries, deduplicates saved references and FP8 components, and records owners and consumers without owning payload tensors. History is bounded and reports truncation. Live-storage counters continue tracking when detailed history reaches its bound.

The existing [measurement harness](../../benchmarks/analysis/measurement.py) and [metadata estimator](../../src/deepopt/memory/estimation.py) use this accounting. Diagnostic dispatch observation is opt-in. Visible tensor peaks, saved-state peaks and CUDA allocator peaks remain separate; native internal workspace is explicitly unknown.

[Capacity admission](../../src/deepopt/memory/capacity.py) requires known current device capacity, incremental tensor demand, declared workspace and a safety margin. The default margin is the greater of 10% of capacity and 1 GiB. Unknowns reject admission. The result retains the requested workload without changing physical batch size or introducing accumulation.

The maintained [VAE audit](../../examples/vae/audit_activation_memory.py) and [local128 target configuration](../../examples/vae/activation_memory_local128.toml) reproduce the requested architecture, image dimensions and codec settings without allocating the real target payload. Read the audit's dtype-policy, optimizer-execution and source-stability fields before interpreting it. An explicitly emulated FP32 GroupNorm policy is not CUDA semantic validation. The configured custom optimizer is not silently replaced when its implementation is unavailable. This ledger is not a complete CUDA training peak or a batch128 fit claim.

```sh
PYTHONPATH=src .venv/bin/python -m examples.vae.audit_activation_memory \
  --output work/vae_local128_metadata.json
```

Use a source snapshot when other sessions are editing the checkout. A source change during the run invalidates the result and yields a nonzero exit status.

## Bounded BF16 kernel configurations

[Configuration metadata](../../src/deepopt/opt/backends/_shared/bf16_gemm.py) supports independent BM/BN/BK, warps, stages, row/column program ordering and grouped M ordering. Bounded candidate tables distinguish forward/dX/dW and include shape, layouts, device capability, codec orientation, epilogue and numerical/source/runtime policy in immutable keys. Caches contain metadata only.

The [shared executor](../../src/deepopt/opt/backends/bf16.py) accepts an explicit `BF16GemmConfig` through `kernel_config`; incompatible native/TileLang requests reject it. Absent an explicit configuration, the previous square launch remains selected. No candidate table is a measured winner and no tuning runs inside the update or capture path. Opt-in GEMM diagnostics distinguish semantic reference, native and generated execution, with available kernel resource metadata.

Packed-read dW remains an experimental explicit control. This change does not qualify a panel-decoding route, split-K, persistent kernels, automatic routing or the historical H100 regression shapes.

## Dense working buffers and dispatch

The [native convolution helper](../../src/deepopt/opt/backends/native/convolution/ops.py) shares input/bias gradient routing between eager and compiler paths. Supported CPU cases use an opposite-convolution adjoint and a qualified direct bias reduction without input-sized descriptors. Geometry, layout, dtype, deterministic and reduction-rounding restrictions retain a visible native fallback. CUDA algorithm equivalence and actual allocation reductions still need device qualification. Padding semantics and shape-only adjoints are supplied by the coordinated native AutoDiff implementation.

The [numerical callback factory](../../src/deepopt/compiler/autodiff/semantic.py) prebinds immutable callsite labels, imports and role identity once per specialization, while retaining dynamic numerical-policy scopes and profiler behavior. Fused derivative GEMMs carry the dX role. No dispatch speedup is claimed without a measured full-update comparison.

## Verification

Focused contracts live in the existing unittest layout, including [schedule/lifetime tests](../../tests/compiler/test_schedule_memory.py), [storage/capacity tests](../../tests/memory/test_storage_ledger.py), [kernel configuration tests](../../tests/opt/test_bf16_gemm_configuration.py), and [spatial allocation tests](../../tests/opt/test_spatial_buffer_allocations.py).

```sh
PYTHONPATH=src .venv/bin/python -m unittest \
  tests.compiler.test_schedule_memory \
  tests.compiler.test_native_autodiff tests.compiler.test_native_autodiff_cuda \
  tests.compiler.test_bf16 tests.compiler.test_fusion_reporting tests.compiler.test_backend \
  tests.memory.test_storage_ledger tests.memory.test_memory_estimation \
  tests.opt.test_bf16_gemm_configuration tests.opt.test_spatial_buffer_allocations \
  tests.opt.test_convolution_native tests.opt.test_convolution_transpose \
  tests.benchmarks.test_bf16_measurement tests.benchmarks.test_bf16_qualification
```

CUDA qualification must separately establish codec-oracle gradients, real launches, resources, working allocations, full-update latency and repeated-step baselines from a stable executed source. Use fresh matched trials and verify the source and recipe for each comparison. A passing CPU suite or metadata audit does not satisfy them.
