# Saved-activation compression

[Documentation index](../README.md) · [FP8 guide](../fp8/README.md) · [Benchmark catalog](../../benchmarks/README.md)

See [performance recovery](performance_recovery.md) for the H100 bottleneck,
tiled unpack/transpose implementation, compatibility fixes and target-GPU
validation commands.

Affine compression reduces the activations retained for linear weight
gradients. Forward still computes with the original FP8 or BF16 activation.
Backward reconstructs the saved activation and uses the existing GEMM backend.
For a linear operation, the weight gradient is approximate; input and bias
gradients keep their usual operands. Attention can also compress its saved
query, key and value tensors, making the corresponding attention gradients
approximate. This is a memory optimization whose speed must be measured on
the target workload.

## Configuration

Activation compression has its own section, separate from FP8:

```toml
[runtime.activation_compression]
enabled = true
group_size = 512
bits = 2
# Optional: attention_bits = 2
```

The FP8 and activation-compression sections are independent:

| FP8 enabled | Compression enabled | Linear execution |
| --- | --- | --- |
| No | No | Native linear with ordinary saved activations |
| No | Yes | Native BF16 linear with packed saved activations |
| Yes | No | FP8 linear with ordinary FP8 saved activations |
| Yes | Yes | FP8 linear with fused casting and activation packing |

The native compression row assumes BF16 compute, selected by native AMP or
BF16 inputs. The Trainer and autocast scope apply this configuration directly;
standalone functional calls use `linear_plan(config=...)` to resolve it once.

It is disabled by default. The default group size is 512 for BF16 and for
tensorwise FP8. A positive `group_size` explicitly selects a different group
size. These groups follow the logical flattened activation matrix, with a
masked final group when necessary.

`bits` controls the global activation format and defaults to 2. Supported
widths are 1, 2, 4 and 8 bits. Attention's saved Q, K and V use
`max(4, bits)` by default, so a globally selected 1-bit or 2-bit format does
not lower Q/K/V below 4 bits. Set `attention_bits` explicitly to override that
floor, including 1-bit or 2-bit attention compression:

```toml
[runtime.activation_compression]
enabled = true
group_size = 512
bits = 2
attention_bits = 1
```

This example compresses projection inputs (`x`) to 2 bits and the attention
operation's saved Q/K/V tensors to 1 bit. The override affects only Q/K/V;
it does not change projection inputs, attention probabilities, masks, dropout
state, attention outputs or logsumexp values. Omit `attention_bits` to retain
the default floor; use `None` in the Python configuration API.

When FP8 activation scaling uses `block_shapes.activation`, compression uses
exactly the same rectangular groups. That rectangle overrides the activation
compression group size. For example:

```toml
[runtime.activation_compression]
enabled = true
group_size = 512

[runtime.fp8]
enabled = true
backend = "triton"
trace_mode = "modules"

[runtime.fp8.block_shapes]
activation = [1, 32]
```

Here each compression group is a 1-by-32 activation scaling block. A 32-by-32
activation scaling block would instead compress 1024 values per group. Edge
blocks exclude padding from their minimum and maximum. Rowwise FP8 weight
gradients retain the existing independently quantized transpose convention;
compression follows its scale blocks in that orientation.

The Trainer and training runtime read the configuration. A manual autocast
scope can specify the same policy explicitly:

```python
import deepopt
import torch
from deepopt.opt.policy import prepare_fp8_model

settings = deepopt.FP8Config(
    enabled=True, backend="triton", trace_mode="modules", intercept_linear=False,
)
model, report = prepare_fp8_model(model, settings)
with deepopt.autocast(
    "cuda", dtype=torch.bfloat16, fp8=settings,
    activation_compression={"enabled": True, "group_size": 512},
    cache_enabled=False, check_finite=False,
):
    loss = loss_fn(model(inputs), targets)
    loss.backward()
```

Standalone projections share the functional entry point for native BF16 and
FP8. Build a plan once and reuse it for each call:

```python
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import linear_plan

plan = linear_plan(
    fp8=False,
    activation_compression={"enabled": True, "group_size": 512},
)

with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
    output = doF.do_linear(inputs, weight, bias, plan=plan)
output.backward(upstream_gradient)
```

Here `fp8=False` selects native matmul, with the affine codec used only to
store activations for the weight gradient. To select tensorwise FP8 instead,
build the plan with `fp8={"enabled": True, "backend": "triton"}`. Both choices
use the same functional call. Stateful delayed FP8 scaling still uses a
prepared model adapter as shown above.

An explicit native plan owns its compute and compression settings even
inside an FP8 autocast scope. Existing explicit FP8 plans retain scoped FP8
and activation-compression overrides. Without a plan, `do_linear` uses the
active DeepOpt policy and otherwise behaves like native linear. Automatic
compression of ordinary `torch.nn.functional.linear` calls is enabled only
when the effective native compute dtype is BF16. The former `compressed_linear`
import remains a compatibility wrapper around the same native-plan dispatch.

## Representation and FP8 integration

Codec functions name both operations explicitly:

| Function | Result |
| --- | --- |
| [`quantize_pack_activation`](../../src/deepopt/opt/quantization/activation.py) | Affine quantization and bit packing into a `PackedActivation` |
| [`dequantize_unpack_activation`](../../src/deepopt/opt/quantization/activation.py) | Code unpacking and dequantization into the requested floating-point dtype |
| [`quantize_pack_fp8_activation`](../../src/deepopt/opt/quantization/activation_fp8.py) | Forward FP8 operand and a quantized, packed backward operand |

Each group stores `32 / bits` codes per `int32` word, plus one FP32 minimum and
one FP32 step. With 512 values per full group this costs 136 bytes, or 2.125
bits per value at the default 2-bit width. Relative to activation values alone, that is approximately
3.76 times smaller than FP8 and 7.53 times smaller than BF16. Partial groups
and small FP8 scale blocks have more padding or metadata overhead. Existing
FP8 inverse scales remain separate and are still needed.

The FP8 path compresses the FP8-rounded values in their scaled domain.
The fused cast-and-pack kernels emit the normal FP8 forward operand and the
packed backward payload. They reuse loaded activation values for packing;
tensorwise current scaling still requires its existing amax reduction.
Groups larger than 16,384 values use bounded partial min/max reductions and
an additional input pass before the fused cast-and-pack launch.
Backward unpacks directly to the required FP8 dtype and orientation, retaining
the original inverse scales. There is no full BF16 reconstruction between
the packed payload and the FP8 GEMM.

Current scaling supports tensorwise and rectangular block scaling. Delayed
scaling supports tensorwise scaling and retains the forward scale snapshot
and normal amax update rules. Delayed scaling still requires eager processing;
activation compression does not remove that existing CUDA Graph restriction.

The BF16 path uses the same affine codec, reconstructs directly to BF16, and
calls native matmul. The implementation does not introduce a 2-bit GEMM.
Packed linear values and affine metadata go through PyTorch saved tensors, so
saved tensor hooks and offloading can observe the payload.

## Lossless boolean masks

Boolean masks have a separate lossless codec, independent of BF16/FP8 and the
lossy activation-compression policy. Its [Triton backend](../../src/deepopt/opt/backends/triton/quantization/activation/boolean.py)
lives alongside the affine, amax and FP8 kernels. Functions use `pack` and
`unpack` because boolean values require no quantization or dequantization.

```python
import torch
from deepopt.opt.quantization.boolean import pack_boolean_mask, unpack_boolean_mask

mask = torch.rand(4, 33, device="cuda") > 0.5
packed = pack_boolean_mask(mask)
restored = unpack_boolean_mask(packed)
torch.testing.assert_close(restored, mask, rtol=0, atol=0)
```

The [public codec](../../src/deepopt/opt/quantization/boolean.py) returns a
`PackedBooleanMask` containing an independent contiguous `int32` tensor and
the original shape. Values follow logical row-major order, with 32 booleans
per word, lowest bit first. Unused high bits of the final word are zero.
There are no scales, minima, group sizes or precision settings. Full words
use 8 times less tensor storage than byte-sized booleans; short tails have
word padding. Python metadata and allocator reservations are excluded.

Packing supports scalar, empty, transposed, sliced, channels-last and broadcast
views. CUDA kernels read source strides directly without materializing a
contiguous input. Unpacking returns an independent contiguous `torch.bool`
tensor of the original shape, without restoring original strides or aliases.
Packing expanded views can increase physical storage: their broadcast source
may already occupy much less memory than the logical tensor.

CPU uses a PyTorch reference implementation. FakeTensor/meta execution only
constructs output metadata and does not import the Triton implementation.
CUDA dispatch is lazy, preserves the input device and current stream, and
supports warmed CUDA Graph capture/replay. Boolean kernels require NVIDIA
CUDA and Triton, without the Hopper FP8 hardware requirement.

This API is a standalone storage primitive. Existing attention Q/K/V
compression, attention masks and dropout saves retain their current behavior;
there is no automatic mask interception or new configuration option.

The installed-library [benchmark](../../benchmarks/activation_compression/benchmark_boolean_mask.py)
measures pack, unpack, roundtrip and a contiguous boolean-copy baseline:

```bash
python benchmarks/activation_compression/benchmark_boolean_mask.py \
  --shape 16x12x197x197 --output work/boolean-mask.json

python benchmarks/activation_compression/benchmark_boolean_mask.py \
  --device cpu --shape 17x33 --iterations 10 --output work/boolean-mask-cpu.json
```

It reports exact reconstruction, payload bytes, strides and timing repeats for
contiguous, transposed, sliced and expanded inputs. Eager timings include output
allocation and host submission; CUDA event intervals include submission gaps.
Neither measurement establishes a full training speedup.

[CPU tests](../../tests/opt/test_boolean_mask.py) cover the format and interpret
the installed JIT bodies for indexing checks. They do not validate CUDA
compilation. Run the [real CUDA tests](../../tests/cuda/test_boolean_mask.py) on
an NVIDIA machine, including graph replay and strided reads:

```bash
DEEPOPT_REQUIRE_CUDA_BOOLEAN=1 python -m unittest tests.cuda.test_boolean_mask -v
```

## Attention Q/K/V compression

Use the same policy for native scaled dot product attention:

```python
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import linear_plan

plan = linear_plan(
    fp8=False,
    activation_compression={"enabled": True, "bits": 2, "attention_bits": 1},
)
output = doF.scaled_dot_product_attention(query, key, value, plan=plan)
output.backward(upstream_gradient)
```

The functional call preserves PyTorch's forward operation and backend
selection, including native fused attention where available. Only the saved
Q/K/V operands are packed and later reconstructed for backward. Softmax
probabilities, masks, dropout state, outputs and logsumexp remain exact;
the forward output is unchanged. Q/K/V gradients and gradients propagated
through their projections can differ because backward uses reconstructed
operands. Test training convergence before selecting a lower bit width.

Within an enabled DeepOpt activation-compression scope, ordinary
`torch.nn.functional.scaled_dot_product_attention` calls use the same policy,
including when FP8 linear interception is disabled. The prepared
`DOMultiheadAttention` adapter applies the policy to both its native SDPA
branch and its explicit attention-weights branch. Projection input compression
continues to use global `bits` independently. Inference and calls without any
required gradient do not pack attention operands.

Attention compression uses a scoped saved-tensor hook. An enclosing saved-tensor
offload or checkpoint hook does not receive the attention operation's saves;
other operations outside that scope retain their normal hook behavior.

## Measurement

Install the updated library into the GPU environment, then copy and run the
standalone benchmark. It imports installed DeepOpt without altering Python's
module search path:

```bash
python benchmark_activation_compression.py \
  --shape 4096x768x3072 --group-size 512 --output compression-tensorwise.json

python benchmark_activation_compression.py \
  --paths fp8 --shape 4096x768x3072 --fp8-block 1x32 \
  --output compression-rowwise.json

python benchmark_activation_compression.py \
  --paths fp8 --scaling-recipe delayed --group-size 512 \
  --output compression-delayed.json
```

The script is [benchmark_activation_compression.py](../../benchmarks/activation_compression/benchmark_activation_compression.py).
It compares compression off/on for each selected path using the same inputs
and FP32 master weights with BF16 AMP. Timed eager steps include forward,
backward, FP8 data preparation, packing, unpacking, and gradient clearing.
They exclude optimizer updates, initialization, warmup, and separate payload
accounting. Both wall time and CUDA stream event intervals include host
submission gaps; neither should be interpreted as summed kernel execution
time.

JSON results include all timing repeats, saved tensor payload bytes, peak
extra CUDA allocation for a separate step, and numerical differences against
compression off. Payload bytes sum logical saved tensor elements; peak
allocation also includes gradients and temporary workspaces. Forward, input
gradient, and bias gradient should agree, while weight-gradient error measures
the effect of the 2-bit codec. Validate convergence separately before using
compression for a full training run.

Real CUDA tests are in [test_activation_compression.py](../../tests/cuda/test_activation_compression.py):

```bash
DEEPOPT_REQUIRE_CUDA_FP8=1 python -m unittest tests.cuda.test_activation_compression -v
```

They compare fused casts and packed payloads with the CPU reference, exercise
strides, tails and constant groups, check reconstructed FP8 weight gradients,
verify saved payloads, and cover current-scaling CUDA Graph replay. They skip
without an NVIDIA GPU with compute capability 9 or newer and Triton.

[Attention tests](../../tests/cuda/test_attention_compression.py) separately
exercise native math and FlashAttention with compressed Q/K/V saves:

```bash
python -m unittest tests.cuda.test_attention_compression -v
```
