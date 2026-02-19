# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_yyu496/wy/cwy3qxyw6imnqvzx5oxclnym6yebmfbnpal67ntph2jmu3my2fca.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_310
# Graph fragment:
#   %triton_kernel_wrapper_mutation_310 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 218, constant_args_idx: 311, grid: [(150528, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %view_1, P_ptr: %empty, S_ptr: %empty_1, M_ptr: %empty_2, stride_x0: 512, stride_x1: 1, stride_p0: 32, stride_p1: 1, BITS: 2, VPW: 16, NWORDS: 32, QMAX: 3}})
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 616562688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 77070336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x1 + 150528*((x0 + 512*((x1 % 294))) // 150528)), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp1, None)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/act_triton_kernel.py:11
quant_pack_kernel_0 = async_compile.triton('quant_pack_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'quant_pack_kernel_0', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_ptr': '*bf16', 'P_ptr': '*i32', 'S_ptr': '*bf16', 'M_ptr': '*bf16', 'stride_x0': 'constexpr', 'stride_x1': 'constexpr', 'stride_p0': 'constexpr', 'stride_p1': 'constexpr', 'BITS': 'constexpr', 'VPW': 'constexpr', 'NWORDS': 'constexpr', 'QMAX': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 32, 'stride_p1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32, 'QMAX': 3}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def quant_pack_kernel(
    X_ptr, P_ptr, S_ptr, M_ptr,
    stride_x0: tl.constexpr, stride_x1: tl.constexpr,
    stride_p0: tl.constexpr, stride_p1: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    QMAX: tl.constexpr,
):
    pid = tl.program_id(0)

    # ---- block pointer: [NWORDS, VPW] = 256 values ----
    x_block_ptr = tl.make_block_ptr(
        base=X_ptr + pid * stride_x0,
        shape=(NWORDS, VPW),
        strides=(stride_x1 * VPW, stride_x1),
        offsets=(0, 0),
        block_shape=(NWORDS, VPW),
        order=(0, 1),
    )

    # ---- single global load ----
    x = tl.load(x_block_ptr).to(tl.float32)   # [NWORDS, VPW]

    # ---- scalar min / max over all 256 values ----
    xmin = tl.min(x, axis=1)      # [NWORDS]
    xmin = tl.min(xmin, axis=0)   # scalar

    xmax = tl.max(x, axis=1)
    xmax = tl.max(xmax, axis=0)

    rng = xmax - xmin

    scale = (rng / tl.full([], QMAX, tl.float32))
    scale = tl.where(rng > 0.0, scale, tl.full([], 1.0, tl.float32))

    tl.store(S_ptr + pid, scale)
    tl.store(M_ptr + pid, xmin)

    inv_scale = tl.full([], 1.0, tl.float32) / scale

    # ---- pack ----
    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    eps  = tl.full([VPW], 1e-6, tl.float32)
    half = tl.full([VPW], 0.5,  tl.float32)

    # quantize all at once
    qf = (x - xmin) * inv_scale + (half - eps)
    qi = qf.to(tl.int32)
    qi = tl.maximum(qi, 0)
    qi = tl.minimum(qi, QMAX)

    # pack words: [NWORDS]
    words = tl.sum(qi << shifts[None, :], axis=1)

    p_block_ptr = tl.make_block_ptr(
        base = P_ptr + pid * stride_p0,
        shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(NWORDS,),
        order=(0,)
    )

    tl.store(p_block_ptr, words.to(tl.int32))

    # tl.store(
    #     P_ptr + pid * stride_p0 + tl.arange(0, NWORDS) * stride_p1,
    #     words
    # )
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jx/cjx5c4m6rifpslk3f5b7ky4cirawyakd6xkm4lsc6npldz76mhgc.py
# Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   conv1 => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 308281344, 'x': 308281344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jg/cjgw6y2lr3knzfvbthlxdauatks7y6bzjxliio3lnlplogyngkj2.py
# Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   conv1 => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 37632, 'x': 37632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 3*x2 + 147*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7u/c7ubxjbrcws4zyxbtayg6nqpywevwynyatltfpe4w4aolsdqg6cg.py
# Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   bn1 => full_default
# Graph fragment:
#   %full_default : [num_users=15] = call_function[target=torch.ops.aten.full.default](args = ([64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_3 = async_compile.triton('triton_poi_fused_zeros_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bv/cbv6hteucmyeo7iqva6f2u7ap5boe734lmfqnkjqctkgdhktwgcn.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_171, clone_default_172, clone_default_175, clone_default_176, clone_default_178, clone_default_179
# Graph fragment:
#   %clone_default_178 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_356,), kwargs = {})
#   %clone_default_179 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_358,), kwargs = {})
#   %clone_default_175 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_350,), kwargs = {})
#   %clone_default_176 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_352,), kwargs = {})
#   %clone_default_171 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_342,), kwargs = {})
#   %clone_default_172 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_344,), kwargs = {})
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3328}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
    tl.store(out_ptr3 + (x0), tmp0, xmask)
    tl.store(out_ptr4 + (x0), tmp0, xmask)
    tl.store(out_ptr5 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/va/cvailrm73gk6fbsun65w5ad7pbftkzinmt72fiphvmnc6xrm2bkk.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_308, triton_kernel_wrapper_mutation_309
# Graph fragment:
#   %triton_kernel_wrapper_mutation_309 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 219, constant_args_idx: 312, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_7, SUM: %as_strided_default_357, SUMSQ: %as_strided_default_359, M: 6422528, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_308 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 220, constant_args_idx: 313, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_7, MEAN: %div, INVSTD: %rsqrt, GAMMA: %primals_4, BETA: %primals_5, Y: %permute, X_hat: %permute_1, M: 6422528, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 802816*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 12544*y3), tmp0, xmask)
    tl.store(out_ptr1 + (x2 + 12544*y3), tmp0, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_1 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_1', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pk/cpkiua3xivk5e4mpfcml7326y52cdjctfngig6dqu2do7uto46zi.py
# Topologically Sorted Source Nodes: [bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_308
#   bn1 => add_1, clamp_min, div, div_1, full_default_2, mul_1, rsqrt, sub
# Graph fragment:
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 6422528.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_357, %full_default_2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_359, %full_default_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %div), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %mul_1), kwargs = {})
#   %clamp_min : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %triton_kernel_wrapper_mutation_308 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 220, constant_args_idx: 313, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_7, MEAN: %div, INVSTD: %rsqrt, GAMMA: %primals_4, BETA: %primals_5, Y: %permute, X_hat: %permute_1, M: 6422528, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_6 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 1.5570192920918366e-07
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_2 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_2', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hs/chsm4rmlm4xelojtggp6dlkqvptayymcftbsioum6ulzwufbpip5.py
# Topologically Sorted Source Nodes: [relu, ], Original ATen: [aten.empty_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_306
#   relu => permute_2
# Graph fragment:
#   %permute_2 : [num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%empty_8, [0, 1, 2, 3]), kwargs = {})
#   %triton_kernel_wrapper_mutation_306 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 315, grid: [(401408, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %view_18, Y_ptr: %permute_2, Mask_prt: %full_default_4, n_elts: 411041792, BLOCK_SIZE: 1024}})
triton_poi_fused_empty_like_7 = async_compile.triton('triton_poi_fused_empty_like_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_empty_like_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2466250752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_empty_like_7(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 12544)
    x1 = ((xindex // 12544) % 64)
    x2 = xindex // 802816
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4c/c4c6pxysrixcho5xmaybu6j2qmobgyi3jrlbvnhvhmhlfdxbhmzy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_306
# Graph fragment:
#   %triton_kernel_wrapper_mutation_306 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 315, grid: [(401408, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %view_18, Y_ptr: %permute_2, Mask_prt: %full_default_4, n_elts: 411041792, BLOCK_SIZE: 1024}})
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:915
relu_kernel_3 = async_compile.triton('relu_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'relu_kernel_3', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_ptr': '*bf16', 'Y_ptr': '*bf16', 'Mask_prt': '*i8', 'n_elts': 'i32', 'BLOCK_SIZE': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_SIZE': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def relu_kernel(
    X_ptr, Y_ptr, Mask_prt,
    n_elts,
    BLOCK_SIZE: tl.constexpr
): 
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elts

    x = tl.load(X_ptr + offs, mask=mask, other=0.)
    mask_relu = x > 0
    y = tl.where(mask_relu, x, 0.)

    tl.store(Y_ptr + offs, y, mask=mask)
    tl.store(Mask_prt + offs, mask_relu.to(tl.int8), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pq/cpqa5j5sroxpivyattbyu52ez3bcl4vxyabgondjgu4m6lm4zkbr.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   relu => full_default_5
# Graph fragment:
#   %full_default_5 : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([802816, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_9 = async_compile.triton('triton_poi_fused_zeros_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/dh/cdhjxaawfegq6pcuh4hh2u4ghwh3fcfahqsi5nny7oo3tul4jclf.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_177
# Graph fragment:
#   %clone_default_177 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_354,), kwargs = {})
triton_poi_fused_10 = async_compile.triton('triton_poi_fused_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 154140672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/act_triton_kernel.py:143
pack_kernel_4 = async_compile.triton('pack_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'pack_kernel_4', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_ptr': '*i8', 'P_ptr': '*i32', 'stride_x0': 'constexpr', 'stride_x1': 'constexpr', 'stride_p0': 'constexpr', 'stride_p1': 'constexpr', 'BITS': 'constexpr', 'VPW': 'constexpr', 'NWORDS': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'stride_x0': 512, 'stride_x1': 1, 'stride_p0': 16, 'stride_p1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def pack_kernel(
    X_ptr, P_ptr,
    stride_x0: tl.constexpr, stride_x1: tl.constexpr,
    stride_p0: tl.constexpr, stride_p1: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr
):
    pid = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base = X_ptr + pid * stride_x0,
        shape=(NWORDS, VPW),
        strides=(stride_x1 * VPW, stride_x1),
        offsets=(0, 0),
        block_shape=(NWORDS, VPW),
        order=(0, 1),
    )

    x = tl.load(x_block_ptr)

    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    words = tl.sum(x << shifts[None, :], axis=1).to(tl.int32)

    # tl.store(P_ptr + pid * stride_p0 + tl.arange(0, NWORDS) * stride_p1,
    #          words)
    p_block_ptr = tl.make_block_ptr(
        base = P_ptr + pid * stride_p0,
        shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(NWORDS,),
        order=(0,)
    )
    tl.store(p_block_ptr, words)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ze/czemvma53ydau2inoorjhbsip456hui4n2dm7udjqepdghbixrv5.py
# Topologically Sorted Source Nodes: [maxpool, layer1_0_conv1, layer1_0_downsample_0], Original ATen: [aten.max_pool2d_with_indices, aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_conv1 => convolution_1
#   layer1_0_downsample_0 => convolution_4
#   maxpool => _low_memory_max_pool_with_offsets, getitem_14
# Graph fragment:
#   %_low_memory_max_pool_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%permute_2, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_14 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 1), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_13, %convert_element_type_2, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_13, %convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_11 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i8', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1027604480, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_11(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex // 56
    x1 = (xindex % 56)
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 64)
    y4 = yindex // 64
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + 2*x1 + 224*x2 + 12544*y0), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-112) + 2*x1 + 224*x2 + 12544*y0), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-111) + 2*x1 + 224*x2 + 12544*y0), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x1 + 224*x2 + 12544*y0), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x1 + 224*x2 + 12544*y0), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x1 + 224*x2 + 12544*y0), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (111 + 2*x1 + 224*x2 + 12544*y0), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (112 + 2*x1 + 224*x2 + 12544*y0), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (113 + 2*x1 + 224*x2 + 12544*y0), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tmp52 = tmp11 > tmp17
    tmp53 = tmp11 == tmp17
    tmp54 = tmp11 != tmp11
    tmp55 = tmp17 != tmp17
    tmp56 = tmp54 > tmp55
    tmp57 = tmp52 | tmp56
    tmp58 = tmp54 & tmp55
    tmp59 = tmp53 | tmp58
    tmp60 = tl.full([1, 1], 1, tl.int64)
    tmp61 = tmp1 < tmp60
    tmp62 = tmp59 & tmp61
    tmp63 = tmp57 | tmp62
    tmp64 = tl.where(tmp63, tmp11, tmp17)
    tmp65 = tl.where(tmp63, tmp1, tmp60)
    tmp66 = tmp64 > tmp24
    tmp67 = tmp64 == tmp24
    tmp68 = tmp64 != tmp64
    tmp69 = tmp24 != tmp24
    tmp70 = tmp68 > tmp69
    tmp71 = tmp66 | tmp70
    tmp72 = tmp68 & tmp69
    tmp73 = tmp67 | tmp72
    tmp74 = tl.full([1, 1], 2, tl.int64)
    tmp75 = tmp65 < tmp74
    tmp76 = tmp73 & tmp75
    tmp77 = tmp71 | tmp76
    tmp78 = tl.where(tmp77, tmp64, tmp24)
    tmp79 = tl.where(tmp77, tmp65, tmp74)
    tmp80 = tmp78 > tmp31
    tmp81 = tmp78 == tmp31
    tmp82 = tmp78 != tmp78
    tmp83 = tmp31 != tmp31
    tmp84 = tmp82 > tmp83
    tmp85 = tmp80 | tmp84
    tmp86 = tmp82 & tmp83
    tmp87 = tmp81 | tmp86
    tmp88 = tl.full([1, 1], 3, tl.int64)
    tmp89 = tmp79 < tmp88
    tmp90 = tmp87 & tmp89
    tmp91 = tmp85 | tmp90
    tmp92 = tl.where(tmp91, tmp78, tmp31)
    tmp93 = tl.where(tmp91, tmp79, tmp88)
    tmp94 = tmp92 > tmp34
    tmp95 = tmp92 == tmp34
    tmp96 = tmp92 != tmp92
    tmp97 = tmp34 != tmp34
    tmp98 = tmp96 > tmp97
    tmp99 = tmp94 | tmp98
    tmp100 = tmp96 & tmp97
    tmp101 = tmp95 | tmp100
    tmp102 = tl.full([1, 1], 4, tl.int64)
    tmp103 = tmp93 < tmp102
    tmp104 = tmp101 & tmp103
    tmp105 = tmp99 | tmp104
    tmp106 = tl.where(tmp105, tmp92, tmp34)
    tmp107 = tl.where(tmp105, tmp93, tmp102)
    tmp108 = tmp106 > tmp37
    tmp109 = tmp106 == tmp37
    tmp110 = tmp106 != tmp106
    tmp111 = tmp37 != tmp37
    tmp112 = tmp110 > tmp111
    tmp113 = tmp108 | tmp112
    tmp114 = tmp110 & tmp111
    tmp115 = tmp109 | tmp114
    tmp116 = tl.full([1, 1], 5, tl.int64)
    tmp117 = tmp107 < tmp116
    tmp118 = tmp115 & tmp117
    tmp119 = tmp113 | tmp118
    tmp120 = tl.where(tmp119, tmp106, tmp37)
    tmp121 = tl.where(tmp119, tmp107, tmp116)
    tmp122 = tmp120 > tmp44
    tmp123 = tmp120 == tmp44
    tmp124 = tmp120 != tmp120
    tmp125 = tmp44 != tmp44
    tmp126 = tmp124 > tmp125
    tmp127 = tmp122 | tmp126
    tmp128 = tmp124 & tmp125
    tmp129 = tmp123 | tmp128
    tmp130 = tl.full([1, 1], 6, tl.int64)
    tmp131 = tmp121 < tmp130
    tmp132 = tmp129 & tmp131
    tmp133 = tmp127 | tmp132
    tmp134 = tl.where(tmp133, tmp120, tmp44)
    tmp135 = tl.where(tmp133, tmp121, tmp130)
    tmp136 = tmp134 > tmp47
    tmp137 = tmp134 == tmp47
    tmp138 = tmp134 != tmp134
    tmp139 = tmp47 != tmp47
    tmp140 = tmp138 > tmp139
    tmp141 = tmp136 | tmp140
    tmp142 = tmp138 & tmp139
    tmp143 = tmp137 | tmp142
    tmp144 = tl.full([1, 1], 7, tl.int64)
    tmp145 = tmp135 < tmp144
    tmp146 = tmp143 & tmp145
    tmp147 = tmp141 | tmp146
    tmp148 = tl.where(tmp147, tmp134, tmp47)
    tmp149 = tl.where(tmp147, tmp135, tmp144)
    tmp150 = tmp148 > tmp50
    tmp151 = tmp148 == tmp50
    tmp152 = tmp148 != tmp148
    tmp153 = tmp50 != tmp50
    tmp154 = tmp152 > tmp153
    tmp155 = tmp150 | tmp154
    tmp156 = tmp152 & tmp153
    tmp157 = tmp151 | tmp156
    tmp158 = tl.full([1, 1], 8, tl.int64)
    tmp159 = tmp149 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tmp155 | tmp160
    tmp162 = tl.where(tmp161, tmp148, tmp50)
    tmp163 = tl.where(tmp161, tmp149, tmp158)
    tmp164 = tmp163.to(tl.int8)
    tl.store(out_ptr0 + (x5 + 3136*y0), tmp51, xmask)
    tl.store(out_ptr1 + (y3 + 64*x5 + 200704*y4), tmp164, xmask)
    tl.store(out_ptr2 + (y3 + 64*x5 + 200704*y4), tmp51, xmask)
    tl.store(out_ptr3 + (y3 + 64*x5 + 200704*y4), tmp51, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/r6/cr6nfhqlmpraeidlel5sftnx7wcmqt3fpfyht6pztu5euby2swrc.py
# Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer1_0_conv1 => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/al/calparmddrv4naeetgsgomii25qxvfnuo6es6b7rnt7vn6jzlh32.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_302, triton_kernel_wrapper_mutation_303
# Graph fragment:
#   %triton_kernel_wrapper_mutation_303 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 223, constant_args_idx: 318, grid: [(64, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_30, SUM: %as_strided_default_351, SUMSQ: %as_strided_default_353, M: 1605632, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_302 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 224, constant_args_idx: 319, grid: [(64, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_30, MEAN: %div_3, INVSTD: %rsqrt_1, GAMMA: %primals_10, BETA: %primals_11, Y: %permute_3, X_hat: %permute_4, M: 1605632, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_13 = async_compile.triton('triton_poi_fused_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_13(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 200704*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp0, xmask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp0, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_5 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_5', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hm/chmaxtyn3sn7hxe3n76reufz7osfzxhonpxbhipd62matgodfk2w.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_302
#   layer1_0_bn1 => add_5, clamp_min_2, div_3, div_4, full_default_8, mul_8, rsqrt_1, sub_2
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_3 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_351, %full_default_8), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_353, %full_default_8), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %div_3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_4, %mul_8), kwargs = {})
#   %clamp_min_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %triton_kernel_wrapper_mutation_302 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 224, constant_args_idx: 319, grid: [(64, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_30, MEAN: %div_3, INVSTD: %rsqrt_1, GAMMA: %primals_10, BETA: %primals_11, Y: %permute_3, X_hat: %permute_4, M: 1605632, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 6.228077168367346e-07
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_6 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_6', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ep/cep7czdv4ay3sixayvctegmeqlunj2bwxdmgpthud6e5e4ef3sjc.py
# Topologically Sorted Source Nodes: [layer1_0_relu], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer1_0_relu => full_default_10
# Graph fragment:
#   %full_default_10 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([512, 64, 56, 56], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_15 = async_compile.triton('triton_poi_fused_zeros_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jx/cjxpvclmrzt3i5jbifcwtiykorcxje2yokczcnburat6d6ed4kyg.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_170, clone_default_174
# Graph fragment:
#   %clone_default_174 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_348,), kwargs = {})
#   %clone_default_170 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_340,), kwargs = {})
triton_poi_fused_16 = async_compile.triton('triton_poi_fused_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 513802240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vn/cvnedggb6aknryv5wg2rpflegq4hiow6r3vyjfy4jmpgflncbdfe.py
# Topologically Sorted Source Nodes: [layer1_0_relu], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer1_0_relu => full_default_11
# Graph fragment:
#   %full_default_11 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([200704, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_17 = async_compile.triton('triton_poi_fused_zeros_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25690112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_17(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ym/cymiuv7fzsb7t245whj2ba4vzgvx7voa5ebel2s7smxgemd4kgdv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_169, clone_default_173
# Graph fragment:
#   %clone_default_173 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_346,), kwargs = {})
#   %clone_default_169 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_338,), kwargs = {})
triton_poi_fused_18 = async_compile.triton('triton_poi_fused_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'out_ptr1': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64225280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_18(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/g5/cg5ox7lp3nv6mjtxbppgwgj457dxybhy7nlfaooulluedhd33xdx.py
# Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer1_0_conv2 => convert_element_type_3
# Graph fragment:
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_14, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_19 = async_compile.triton('triton_poi_fused__to_copy_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 147456, 'x': 147456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yw/cywkzraflnmvphvalp77via4jb66udzenazphf3iyt3poyfn2xu3.py
# Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_conv2 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_5, %convert_element_type_3, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 64*x2 + 200704*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mi/cmixrmlgoeyl4km6zf32ygbriiziyoubwfu6rtzkkue6qu5fhgpo.py
# Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer1_0_conv3 => convert_element_type_4
# Graph fragment:
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_20, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_21 = async_compile.triton('triton_poi_fused__to_copy_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qh/cqhuamyyorxjru46c2gjacrsq6657ivelpekjf7i2ridtzuwtynu.py
# Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer1_0_bn3 => full_default_18
# Graph fragment:
#   %full_default_18 : [num_users=33] = call_function[target=torch.ops.aten.full.default](args = ([256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_22 = async_compile.triton('triton_poi_fused_zeros_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_22(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jq/cjqjg3tpefgkdtrppeh23ku372rgbk7epoqwb7kcnpo7rjicrdfz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_165, clone_default_166, clone_default_167, clone_default_168
# Graph fragment:
#   %clone_default_167 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_334,), kwargs = {})
#   %clone_default_168 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_336,), kwargs = {})
#   %clone_default_165 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_330,), kwargs = {})
#   %clone_default_166 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_332,), kwargs = {})
triton_poi_fused_23 = async_compile.triton('triton_poi_fused_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_23(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
    tl.store(out_ptr3 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5e/c5epe4gcoya32x3ybnwkkmziuqtg6gqkphoence5ae4zcjrjgg23.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_290, triton_kernel_wrapper_mutation_291
# Graph fragment:
#   %triton_kernel_wrapper_mutation_291 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 231, constant_args_idx: 330, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_80, SUM: %as_strided_default_335, SUMSQ: %as_strided_default_337, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_290 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 232, constant_args_idx: 331, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_80, MEAN: %div_9, INVSTD: %rsqrt_3, GAMMA: %primals_22, BETA: %primals_23, Y: %permute_9, X_hat: %permute_10, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_24 = async_compile.triton('triton_poi_fused_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_24(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 3136
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 802816*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_7 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_7', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ao/caoj7bcrkleq2jhfo6s5da2k66lgnygjwqhgy4tzwiooe2anvyjo.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_290
#   layer1_0_bn1 => full_default_8
#   layer1_0_bn3 => add_13, clamp_min_6, div_10, div_9, mul_22, rsqrt_3, sub_6
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_9 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_335, %full_default_8), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_337, %full_default_8), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %div_9), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_10, %mul_22), kwargs = {})
#   %clamp_min_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_6, 0.0), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %triton_kernel_wrapper_mutation_290 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 232, constant_args_idx: 331, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_80, MEAN: %div_9, INVSTD: %rsqrt_3, GAMMA: %primals_22, BETA: %primals_23, Y: %permute_9, X_hat: %permute_10, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_25 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_25(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 6.228077168367346e-07
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_8 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_8', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ut/cutnqoxvxrwhsiigme7yxrrdqhqcvavz7rkgawphvjxapsuxhzzf.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_164
# Graph fragment:
#   %clone_default_164 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_328,), kwargs = {})
triton_poi_fused_26 = async_compile.triton('triton_poi_fused_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1233125376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ur/cur24ouuki3nhkkbuo63ivri6jjl6sluo7vagdrvkx5vyij7ckex.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_284
# Graph fragment:
#   %triton_kernel_wrapper_mutation_284 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 337, grid: [(401408, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %add_20, Y_ptr: %permute_13, Mask_prt: %as_strided_default_329, n_elts: 411041792, BLOCK_SIZE: 1024}})
triton_poi_fused_27 = async_compile.triton('triton_poi_fused_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/44/c44rs5jf4ractkbgebqzvwsdvm4dtnwe3b6zrkkmehwkxukqcujw.py
# Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer1_1_conv1 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_13, %convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_28 = async_compile.triton('triton_poi_fused_convolution_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 3136
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 802816*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ie/ciejk55mw7ss4xrdbyzybfctxzgagdvuvkcxrh5hcms4umehqqo5.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_157, clone_default_158, clone_default_161, clone_default_162
# Graph fragment:
#   %clone_default_161 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_322,), kwargs = {})
#   %clone_default_162 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_324,), kwargs = {})
#   %clone_default_157 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_314,), kwargs = {})
#   %clone_default_158 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_316,), kwargs = {})
triton_poi_fused_29 = async_compile.triton('triton_poi_fused_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_29(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
    tl.store(out_ptr3 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/du/cdupdx5o3ao55fx6whsmydst4dglji2lsfr5d7eg3v5ezr7dr2bf.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_153, clone_default_154
# Graph fragment:
#   %clone_default_153 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_306,), kwargs = {})
#   %clone_default_154 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_308,), kwargs = {})
triton_poi_fused_30 = async_compile.triton('triton_poi_fused_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_30(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qh/cqh5jyjxfzgqtp4vcla5lxoxuevg5wtielpsh6mpc2wt4qohzof6.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_146, clone_default_149, clone_default_150
# Graph fragment:
#   %clone_default_149 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_298,), kwargs = {})
#   %clone_default_150 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_300,), kwargs = {})
#   %clone_default_146 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_292,), kwargs = {})
triton_poi_fused_31 = async_compile.triton('triton_poi_fused_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_31(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ne/cnebyzmldjxpdavu5msnycfmng2w7gkfplvxhlr7xbnuylhukobv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_148
# Graph fragment:
#   %clone_default_148 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_296,), kwargs = {})
triton_poi_fused_32 = async_compile.triton('triton_poi_fused_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 308281344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/d7/cd7wd26dwbqzlciojbcrcq73tuqu23prxon7zt4jrhinurvgvm6s.py
# Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_conv1 => convert_element_type_12
# Graph fragment:
#   %convert_element_type_12 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_68, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_33 = async_compile.triton('triton_poi_fused__to_copy_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/id/cidu7swsr34plahoedtbbgjf65wrtmodzuw4qyntnds3tkyspbif.py
# Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_bn1 => full_default_64
# Graph fragment:
#   %full_default_64 : [num_users=17] = call_function[target=torch.ops.aten.full.default](args = ([128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_34 = async_compile.triton('triton_poi_fused_zeros_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_34(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/kk/ckkevxnk5rzu5ludtj6htyflhcm2r3vufzjzfkyikxghnslzpjrp.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_138, clone_default_139, clone_default_141, clone_default_142
# Graph fragment:
#   %clone_default_141 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_282,), kwargs = {})
#   %clone_default_142 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_284,), kwargs = {})
#   %clone_default_138 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_276,), kwargs = {})
#   %clone_default_139 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_278,), kwargs = {})
triton_poi_fused_35 = async_compile.triton('triton_poi_fused_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_35(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
    tl.store(out_ptr3 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2m/c2mz7bgsji4k64iea6sessa6zwi45ycpuc7ny4oy3nw2jbil6jxs.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_244, triton_kernel_wrapper_mutation_245
# Graph fragment:
#   %triton_kernel_wrapper_mutation_245 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 263, constant_args_idx: 376, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_274, SUM: %as_strided_default_283, SUMSQ: %as_strided_default_285, M: 1605632, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_244 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 264, constant_args_idx: 377, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_274, MEAN: %div_33, INVSTD: %rsqrt_11, GAMMA: %primals_70, BETA: %primals_71, Y: %permute_32, X_hat: %permute_33, M: 1605632, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_36 = async_compile.triton('triton_poi_fused_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_36(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 3136
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_9 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_9', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3f/c3fmtoteicebqyahmlnmqre7b5jnigvime6yax7cduemgoej5b3j.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer2_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_244
#   layer1_0_bn1 => full_default_8
#   layer2_0_bn1 => add_48, clamp_min_22, div_33, div_34, mul_78, rsqrt_11, sub_22
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_33 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_283, %full_default_8), kwargs = {})
#   %div_34 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_285, %full_default_8), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_33, %div_33), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_34, %mul_78), kwargs = {})
#   %clamp_min_22 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_22, 0.0), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_22, 1e-05), kwargs = {})
#   %rsqrt_11 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_48,), kwargs = {})
#   %triton_kernel_wrapper_mutation_244 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 264, constant_args_idx: 377, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_274, MEAN: %div_33, INVSTD: %rsqrt_11, GAMMA: %primals_70, BETA: %primals_71, Y: %permute_32, X_hat: %permute_33, M: 1605632, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_37 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_37(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 6.228077168367346e-07
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_10 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_10', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/rb/crbzcpnt7ndc56rnuqhgt4aprkknftdnph2r53ibakymzuftkrsx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_242
# Graph fragment:
#   %triton_kernel_wrapper_mutation_242 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 379, grid: [(200704, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %view_285, Y_ptr: %permute_34, Mask_prt: %full_default_68, n_elts: 205520896, BLOCK_SIZE: 1024}})
triton_poi_fused_38 = async_compile.triton('triton_poi_fused_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_38(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/np/cnpbiymq3i6l44cybnf62kpr2s3plslckwcjvwfy6on4dwft5ipf.py
# Topologically Sorted Source Nodes: [layer2_0_relu], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_relu => full_default_69
# Graph fragment:
#   %full_default_69 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([401408, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_39 = async_compile.triton('triton_poi_fused_zeros_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51380224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_39(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mv/cmv4nr55fanljw367jcdkgfbak6vbflm42zmhcujcxxgljrkjyax.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_140
# Graph fragment:
#   %clone_default_140 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_280,), kwargs = {})
triton_poi_fused_40 = async_compile.triton('triton_poi_fused_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 77070336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_40(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ny/cnyq6eooxvuq5yjkwkmjnmo4snseikgqe4ual72tstdklmcwvyr2.py
# Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_conv2 => convert_element_type_13
# Graph fragment:
#   %convert_element_type_13 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_74, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_41 = async_compile.triton('triton_poi_fused__to_copy_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 589824, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hn/chnvn62ncyluwkjtcifmczro55pn5zvl37rz3ovnaf7j3qehe5fo.py
# Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_conv2 => convolution_12
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_34, %convert_element_type_13, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_42 = async_compile.triton('triton_poi_fused_convolution_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_42(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 3136
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 401408*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bb/cbb2nachu5uj2vqzfpr5a34lj55qjh36pwuezibdmeucalntk4pw.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_238, triton_kernel_wrapper_mutation_239
# Graph fragment:
#   %triton_kernel_wrapper_mutation_239 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 267, constant_args_idx: 382, grid: [(128, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_299, SUM: %as_strided_default_277, SUMSQ: %as_strided_default_279, M: 401408, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_238 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 268, constant_args_idx: 383, grid: [(128, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_299, MEAN: %div_36, INVSTD: %rsqrt_12, GAMMA: %primals_76, BETA: %primals_77, Y: %permute_35, X_hat: %permute_36, M: 401408, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_43 = async_compile.triton('triton_poi_fused_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 102760448, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_43(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 784
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 100352*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 784*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_11 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_11', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ji/cjizqkfjjtphvylphh4hpstzafbqt7yeo2ehuodgvzq3p6my7kb4.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_238
#   layer2_0_bn2 => add_52, clamp_min_24, div_36, div_37, full_default_72, mul_85, rsqrt_12, sub_24
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_36 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_277, %full_default_72), kwargs = {})
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_279, %full_default_72), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_36, %div_36), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_37, %mul_85), kwargs = {})
#   %clamp_min_24 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0.0), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_24, 1e-05), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_52,), kwargs = {})
#   %triton_kernel_wrapper_mutation_238 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 268, constant_args_idx: 383, grid: [(128, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_299, MEAN: %div_36, INVSTD: %rsqrt_12, GAMMA: %primals_76, BETA: %primals_77, Y: %permute_35, X_hat: %permute_36, M: 401408, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 2.4912308673469386e-06
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_12 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_12', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gp/cgps5i6i3w3tm5tjn2we45u5mcbof2tb4sws4zg3uszlqlvqx6pt.py
# Topologically Sorted Source Nodes: [layer2_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_relu_1 => full_default_74
# Graph fragment:
#   %full_default_74 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([512, 128, 28, 28], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_45 = async_compile.triton('triton_poi_fused_zeros_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_45(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5b/c5bemcvxsax2lzuhwsugv4i5gdb7osneeqe5to6zbhk7scuy3lj7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_137
# Graph fragment:
#   %clone_default_137 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_274,), kwargs = {})
triton_poi_fused_46 = async_compile.triton('triton_poi_fused_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 154140672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/st/cstsrey4eiupg2mhawu77djiojwv6ac3vxw5vhdw57fthkbolwzq.py
# Topologically Sorted Source Nodes: [layer2_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_relu_1 => full_default_75
# Graph fragment:
#   %full_default_75 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([100352, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_47 = async_compile.triton('triton_poi_fused_zeros_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12845056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_47(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ch/cchknqqag2n7vhxa26xmoruhvt5qa5jlmn37ir5yl54y6mwnx6oj.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_136
# Graph fragment:
#   %clone_default_136 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_272,), kwargs = {})
triton_poi_fused_48 = async_compile.triton('triton_poi_fused_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 19267584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pl/cplroci3metvghtfbn4mexdhoubuv2qjektyvh5lfrxes5qkh4sb.py
# Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_conv3 => convert_element_type_14
# Graph fragment:
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_80, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_49 = async_compile.triton('triton_poi_fused__to_copy_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/g7/cg7tejjwkcikw5l43uw5yciqfnanoswz3z7bzieumshxl2e74iid.py
# Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_conv3 => convolution_13
# Graph fragment:
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_37, %convert_element_type_14, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_50 = async_compile.triton('triton_poi_fused_convolution_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_50(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 784
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 100352*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/a7/ca7cvwqxdmwcsh4b2iigcqk4uyie7jzesu7bltdvevyw6bj6yt2g.py
# Topologically Sorted Source Nodes: [layer2_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_bn3 => full_default_76
# Graph fragment:
#   %full_default_76 : [num_users=23] = call_function[target=torch.ops.aten.full.default](args = ([512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_51 = async_compile.triton('triton_poi_fused_zeros_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_51(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pe/cpe4gfpuy3eghxukwetxewbqn3hkagvehq3pa45oqb3g2kglkezx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_132, clone_default_133, clone_default_134, clone_default_135
# Graph fragment:
#   %clone_default_134 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_268,), kwargs = {})
#   %clone_default_135 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_270,), kwargs = {})
#   %clone_default_132 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_264,), kwargs = {})
#   %clone_default_133 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_266,), kwargs = {})
triton_poi_fused_52 = async_compile.triton('triton_poi_fused_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_52(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
    tl.store(out_ptr3 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ep/cepo6ra5spcedsg76dfouox32b3ztomab4mczltqdfnszrwr4ptp.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_232, triton_kernel_wrapper_mutation_233
# Graph fragment:
#   %triton_kernel_wrapper_mutation_233 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 271, constant_args_idx: 388, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_324, SUM: %as_strided_default_269, SUMSQ: %as_strided_default_271, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_232 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 272, constant_args_idx: 389, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_324, MEAN: %div_39, INVSTD: %rsqrt_13, GAMMA: %primals_82, BETA: %primals_83, Y: %permute_38, X_hat: %permute_39, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_53 = async_compile.triton('triton_poi_fused_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_53(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 784
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 784*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_13 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_13', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/re/cre2zkmehxth7bara45ckwoapbcssrfsadynzyouhx6jtkipamj2.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_232
#   layer2_0_bn2 => full_default_72
#   layer2_0_bn3 => add_56, clamp_min_26, div_39, div_40, mul_92, rsqrt_13, sub_26
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_39 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_269, %full_default_72), kwargs = {})
#   %div_40 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_271, %full_default_72), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_39, %div_39), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_40, %mul_92), kwargs = {})
#   %clamp_min_26 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_26, 0.0), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_26, 1e-05), kwargs = {})
#   %rsqrt_13 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_56,), kwargs = {})
#   %triton_kernel_wrapper_mutation_232 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 272, constant_args_idx: 389, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_324, MEAN: %div_39, INVSTD: %rsqrt_13, GAMMA: %primals_82, BETA: %primals_83, Y: %permute_38, X_hat: %permute_39, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_54 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_54(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 2.4912308673469386e-06
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_14 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_14', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/dl/cdlsqrdiobc6czirjxcrhgderq62hcn6skohayiqnisstpjgbhnr.py
# Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_downsample_0 => convert_element_type_15
# Graph fragment:
#   %convert_element_type_15 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_86, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_55 = async_compile.triton('triton_poi_fused__to_copy_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_55(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vt/cvtexauzx3qt5lzki76v3zfmbwoxrk5sjqzytxb2wfhntzd4eart.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_131
# Graph fragment:
#   %clone_default_131 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_262,), kwargs = {})
triton_poi_fused_56 = async_compile.triton('triton_poi_fused_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 616562688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_56(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/zw/czw2egbgeqj7m5w4lbyi2birv5rfe26hyhybfz7cd5m4a3a3g4rk.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_226
# Graph fragment:
#   %triton_kernel_wrapper_mutation_226 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 395, grid: [(200704, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %add_63, Y_ptr: %permute_42, Mask_prt: %as_strided_default_263, n_elts: 205520896, BLOCK_SIZE: 1024}})
triton_poi_fused_57 = async_compile.triton('triton_poi_fused_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_57(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/g6/cg64dsg4x3otjvfkkpn7lxaeaaqapz2njii6nm27coxatpkqhsf3.py
# Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_1_conv1 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_42, %convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_58 = async_compile.triton('triton_poi_fused_convolution_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_58(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 784
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 401408*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/le/cledzkznowm7tde7fjkrpqcoc3wfujl2jvkyzeg3guwg5ux4ollp.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_123, clone_default_127
# Graph fragment:
#   %clone_default_127 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_254,), kwargs = {})
#   %clone_default_123 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_246,), kwargs = {})
triton_poi_fused_59 = async_compile.triton('triton_poi_fused_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256901120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_59(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ep/cepxxw7hktvnhhfacffwx5niiom4bwt3fyuw6loqjyey3axsyysg.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_122, clone_default_126
# Graph fragment:
#   %clone_default_126 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_252,), kwargs = {})
#   %clone_default_122 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_244,), kwargs = {})
triton_poi_fused_60 = async_compile.triton('triton_poi_fused_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'out_ptr1': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_60', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32112640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_60(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qn/cqnpufi3uc2urtdvhkcmuhkft4kxvdiyv77zz5utlkgzmurp4j5s.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_120, clone_default_121
# Graph fragment:
#   %clone_default_120 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_240,), kwargs = {})
#   %clone_default_121 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_242,), kwargs = {})
triton_poi_fused_61 = async_compile.triton('triton_poi_fused_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_61(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4d/c4dbbios7e4pph5len5t6etvi7d7bbmiwmogquj3iecpsiu66ezb.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_101, clone_default_104, clone_default_105
# Graph fragment:
#   %clone_default_104 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_208,), kwargs = {})
#   %clone_default_105 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_210,), kwargs = {})
#   %clone_default_101 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_202,), kwargs = {})
triton_poi_fused_62 = async_compile.triton('triton_poi_fused_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_62(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vd/cvdpaxtw5lgzuefyvuoa65igvcdfppp6p7yx7bzbpvbxx5nmv3aa.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_168, triton_kernel_wrapper_mutation_169
# Graph fragment:
#   %triton_kernel_wrapper_mutation_169 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 315, constant_args_idx: 452, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_595, SUM: %as_strided_default_193, SUMSQ: %as_strided_default_195, M: 401408, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_168 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 316, constant_args_idx: 453, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_595, MEAN: %div_72, INVSTD: %rsqrt_24, GAMMA: %primals_148, BETA: %primals_149, Y: %permute_70, X_hat: %permute_71, M: 401408, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_63 = async_compile.triton('triton_poi_fused_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_63(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 784
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 784*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_15 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_15', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xk/cxk7td4igjihht5sqh5l6ii2srjakgjcfb4ezhgwmhyw54kco4ry.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer3_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_168
#   layer2_0_bn2 => full_default_72
#   layer3_0_bn1 => add_104, clamp_min_48, div_72, div_73, mul_169, rsqrt_24, sub_48
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_72 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_193, %full_default_72), kwargs = {})
#   %div_73 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_195, %full_default_72), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_72, %div_72), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_73, %mul_169), kwargs = {})
#   %clamp_min_48 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_48, 0.0), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_48, 1e-05), kwargs = {})
#   %rsqrt_24 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_104,), kwargs = {})
#   %triton_kernel_wrapper_mutation_168 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 316, constant_args_idx: 453, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_595, MEAN: %div_72, INVSTD: %rsqrt_24, GAMMA: %primals_148, BETA: %primals_149, Y: %permute_70, X_hat: %permute_71, M: 401408, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_64 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_64(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 2.4912308673469386e-06
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_16 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_16', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/zq/czqlgujsyvlr7cfa2r5koto3oqnfq5kgmefcylaxjutkcripcyor.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_95
# Graph fragment:
#   %clone_default_95 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_190,), kwargs = {})
triton_poi_fused_65 = async_compile.triton('triton_poi_fused_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_65', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 38535168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_65(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3z/c3zwf3r4t7uso6akenfr5ojnmp4ruwaau3tv22lxp3fil6bc6y7f.py
# Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer3_0_conv2 => convert_element_type_26
# Graph fragment:
#   %convert_element_type_26 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_152, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_66 = async_compile.triton('triton_poi_fused__to_copy_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_66', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_66(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/27/c27bdnidn3s7wab6f2fegdrbrhsbkreri4abr2rkaevezxol66uy.py
# Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_conv2 => convolution_25
# Graph fragment:
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_72, %convert_element_type_26, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_67 = async_compile.triton('triton_poi_fused_convolution_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_67', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_67(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 784
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 200704*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/d5/cd5mv63kaog2rm55z5qewz4njapway5o5zwaj6vukbh2u5zqv7kj.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_162, triton_kernel_wrapper_mutation_163
# Graph fragment:
#   %triton_kernel_wrapper_mutation_163 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 319, constant_args_idx: 458, grid: [(256, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_620, SUM: %as_strided_default_187, SUMSQ: %as_strided_default_189, M: 100352, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_162 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 320, constant_args_idx: 459, grid: [(256, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_620, MEAN: %div_75, INVSTD: %rsqrt_25, GAMMA: %primals_154, BETA: %primals_155, Y: %permute_73, X_hat: %permute_74, M: 100352, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_68 = async_compile.triton('triton_poi_fused_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_68', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 51380224, 'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_68(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 196
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 50176*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 196*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_17 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_17', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/k5/ck5zjb6amvuo7z3wnhwuai5q5nlyt5rck2mwc3vj5wbig7jd7tqk.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_162
#   layer3_0_bn2 => add_108, clamp_min_50, div_75, div_76, full_default_148, mul_176, rsqrt_25, sub_50
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_75 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_187, %full_default_148), kwargs = {})
#   %div_76 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_189, %full_default_148), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_75, %div_75), kwargs = {})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_76, %mul_176), kwargs = {})
#   %clamp_min_50 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_50, 0.0), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_50, 1e-05), kwargs = {})
#   %rsqrt_25 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_108,), kwargs = {})
#   %triton_kernel_wrapper_mutation_162 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 320, constant_args_idx: 459, grid: [(256, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_620, MEAN: %div_75, INVSTD: %rsqrt_25, GAMMA: %primals_154, BETA: %primals_155, Y: %permute_73, X_hat: %permute_74, M: 100352, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 9.964923469387754e-06
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_18 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_18', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/aj/cajttmndydsaexb3bwopehnub35atvgx4oycwhxdpckopwz5zgim.py
# Topologically Sorted Source Nodes: [layer3_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer3_0_relu_1 => full_default_150
# Graph fragment:
#   %full_default_150 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([512, 256, 14, 14], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_70 = async_compile.triton('triton_poi_fused_zeros_70', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_70', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51380224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_70(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/b7/cb7tbz33tzgqyemmtzm6bcb4w6sgmek6i6cag27hsngisrr46fsq.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_92
# Graph fragment:
#   %clone_default_92 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_184,), kwargs = {})
triton_poi_fused_71 = async_compile.triton('triton_poi_fused_71', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_71', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 77070336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_71(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jy/cjyc62omk523yclbhl5dn6fwcz2y75manyggec5eroxtqmbzfw7n.py
# Topologically Sorted Source Nodes: [layer3_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer3_0_relu_1 => full_default_151
# Graph fragment:
#   %full_default_151 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([50176, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_72 = async_compile.triton('triton_poi_fused_zeros_72', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_72', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6422528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_72(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/om/comtwhu4cc7zspjkukogxy7gv47xyltfsb77mkh673mfjxknf7v7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_91
# Graph fragment:
#   %clone_default_91 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_182,), kwargs = {})
triton_poi_fused_73 = async_compile.triton('triton_poi_fused_73', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_73', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9633792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_73(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bj/cbjpsmy3ryygyzslyilfpk6d5hxdbuvjqyfmbiscucdhialfnxux.py
# Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer3_0_conv3 => convert_element_type_27
# Graph fragment:
#   %convert_element_type_27 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_158, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_74 = async_compile.triton('triton_poi_fused__to_copy_74', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_74', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_74(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/57/c57lw3hrex6ovlhjeft32su3q4ag2kia3go5bkbqwqmgp64u7v4s.py
# Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_conv3 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_75, %convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_75 = async_compile.triton('triton_poi_fused_convolution_75', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_75', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 102760448, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_75(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 196
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 50176*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qg/cqg2cdw3ijixopeoptscnbvtb6axtsrizbjmat45h2dngach2ypa.py
# Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer3_0_bn3 => full_default_152
# Graph fragment:
#   %full_default_152 : [num_users=15] = call_function[target=torch.ops.aten.full.default](args = ([1024], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_76 = async_compile.triton('triton_poi_fused_zeros_76', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_76', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_76(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mu/cmug4fdwyleyoddm27p2pzeu4lt7pvmqlpqbrvh3qu75e5amff65.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_87, clone_default_88, clone_default_89, clone_default_90
# Graph fragment:
#   %clone_default_89 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_178,), kwargs = {})
#   %clone_default_90 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_180,), kwargs = {})
#   %clone_default_87 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_174,), kwargs = {})
#   %clone_default_88 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_176,), kwargs = {})
triton_poi_fused_77 = async_compile.triton('triton_poi_fused_77', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_77', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_77(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
    tl.store(out_ptr3 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2e/c2erty4vefakl6yzi5eozyvvkasaf6asb2kr4fwabugr4zdawjam.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_156, triton_kernel_wrapper_mutation_157
# Graph fragment:
#   %triton_kernel_wrapper_mutation_157 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 323, constant_args_idx: 464, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_645, SUM: %as_strided_default_179, SUMSQ: %as_strided_default_181, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_156 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 324, constant_args_idx: 465, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_645, MEAN: %div_78, INVSTD: %rsqrt_26, GAMMA: %primals_160, BETA: %primals_161, Y: %permute_76, X_hat: %permute_77, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_78 = async_compile.triton('triton_poi_fused_78', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_78', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_78(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 196
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 196*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_19 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_19', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wm/cwmygol25ejxcfkj4qprcc3cqydpdpqqqy7mr4mhwbwhogs2vc5j.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_156
#   layer3_0_bn2 => full_default_148
#   layer3_0_bn3 => add_112, clamp_min_52, div_78, div_79, mul_183, rsqrt_26, sub_52
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_78 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_179, %full_default_148), kwargs = {})
#   %div_79 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_181, %full_default_148), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_78, %div_78), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_79, %mul_183), kwargs = {})
#   %clamp_min_52 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_52, 0.0), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_52, 1e-05), kwargs = {})
#   %rsqrt_26 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_112,), kwargs = {})
#   %triton_kernel_wrapper_mutation_156 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 324, constant_args_idx: 465, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_645, MEAN: %div_78, INVSTD: %rsqrt_26, GAMMA: %primals_160, BETA: %primals_161, Y: %permute_76, X_hat: %permute_77, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 9.964923469387754e-06
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_20 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_20', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/co/ccodl2ust42khbllizewke6vcpghddpnao73gjxc5hguyay3vxrq.py
# Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer3_0_downsample_0 => convert_element_type_28
# Graph fragment:
#   %convert_element_type_28 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_164, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_80 = async_compile.triton('triton_poi_fused__to_copy_80', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_80', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_80(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/dv/cdvspmwi5ap6nz54fpjyexubarkmfzp4p6lylebjp3dgcrxgnumh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_150
# Graph fragment:
#   %triton_kernel_wrapper_mutation_150 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 471, grid: [(100352, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %add_119, Y_ptr: %permute_80, Mask_prt: %as_strided_default_173, n_elts: 102760448, BLOCK_SIZE: 1024}})
triton_poi_fused_81 = async_compile.triton('triton_poi_fused_81', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_81', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_81(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yq/cyqaeyiyszpa2aknkhtr7laajubsipzihb75cx7krmkf3fqrcdy2.py
# Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_1_conv1 => convolution_28
# Graph fragment:
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_80, %convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_82 = async_compile.triton('triton_poi_fused_convolution_82', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_82', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_82(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 196
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 1024*x2 + 200704*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/za/cza2zgxwurd7g4chzw5bryxym4bijcprd2mweesw7nxhv2asaf6m.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_78, clone_default_82
# Graph fragment:
#   %clone_default_82 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_164,), kwargs = {})
#   %clone_default_78 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_156,), kwargs = {})
triton_poi_fused_83 = async_compile.triton('triton_poi_fused_83', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_83', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 128450560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_83(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jf/cjft2lqlzgh5pbi6l5agb74ykyrr5sn56e7jlii6sqv2uhkwtlqv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_77, clone_default_81
# Graph fragment:
#   %clone_default_81 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_162,), kwargs = {})
#   %clone_default_77 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_154,), kwargs = {})
triton_poi_fused_84 = async_compile.triton('triton_poi_fused_84', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'out_ptr1': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_84', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16056320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_84(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4s/c4stv37p4v7edods3t7aw625gtg3ujrs3pjve42o3xt6uigbpswz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_75, clone_default_76
# Graph fragment:
#   %clone_default_75 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_150,), kwargs = {})
#   %clone_default_76 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_152,), kwargs = {})
triton_poi_fused_85 = async_compile.triton('triton_poi_fused_85', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_85', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_85(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/t5/ct5hnrtuytbyde5lq2gqhdnn3aqgrxsba634ytjgbj63mphqa2es.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_32, clone_default_35, clone_default_36
# Graph fragment:
#   %clone_default_35 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_70,), kwargs = {})
#   %clone_default_36 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_72,), kwargs = {})
#   %clone_default_32 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_64,), kwargs = {})
triton_poi_fused_86 = async_compile.triton('triton_poi_fused_86', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_86', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_86(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3z/c3zkh2z5gu65i4ysiybfkkhhlpxvvp4wki62qtli4a7ks5go7mf3.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_31
# Graph fragment:
#   %clone_default_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_62,), kwargs = {})
triton_poi_fused_87 = async_compile.triton('triton_poi_fused_87', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_87', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_87(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/d6/cd6muvbmokr4msvlj642oqfs4z6nrsshtvbmxvurbe67kpkrcdqy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_56, triton_kernel_wrapper_mutation_57
# Graph fragment:
#   %triton_kernel_wrapper_mutation_57 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 391, constant_args_idx: 564, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1066, SUM: %as_strided_default_59, SUMSQ: %as_strided_default_61, M: 100352, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_56 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 392, constant_args_idx: 565, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1066, MEAN: %div_129, INVSTD: %rsqrt_43, GAMMA: %primals_262, BETA: %primals_263, Y: %permute_126, X_hat: %permute_127, M: 100352, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_88 = async_compile.triton('triton_poi_fused_88', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_88', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 102760448, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_88(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 196
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 100352*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 196*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_21 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_21', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ly/clydn74oew2rajnkvj64ekgoucdgt6uvbgkk7qhklp4rkljsvgop.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer4_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_56
#   layer3_0_bn2 => full_default_148
#   layer4_0_bn1 => add_186, clamp_min_86, div_129, div_130, mul_302, rsqrt_43, sub_86
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_129 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_59, %full_default_148), kwargs = {})
#   %div_130 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_61, %full_default_148), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_129, %div_129), kwargs = {})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_130, %mul_302), kwargs = {})
#   %clamp_min_86 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_86, 0.0), kwargs = {})
#   %add_186 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_86, 1e-05), kwargs = {})
#   %rsqrt_43 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_186,), kwargs = {})
#   %triton_kernel_wrapper_mutation_56 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 392, constant_args_idx: 565, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1066, MEAN: %div_129, INVSTD: %rsqrt_43, GAMMA: %primals_262, BETA: %primals_263, Y: %permute_126, X_hat: %permute_127, M: 100352, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_89 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_89', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_89', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_89(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 9.964923469387754e-06
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_22 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_22', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/nk/cnkn7dvxyrhkzu75in3zsqy4sgmwbca6xnbt2vyc27lmasvudfvb.py
# Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer4_0_conv2 => convert_element_type_45
# Graph fragment:
#   %convert_element_type_45 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_266, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_90 = async_compile.triton('triton_poi_fused__to_copy_90', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_90', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9437184, 'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_90(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5k/c5kt67u5chs2jhibje2v4xzy43xcxobw2h4kmc75bhgsb6fheme7.py
# Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_conv2 => convolution_44
# Graph fragment:
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_128, %convert_element_type_45, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_91 = async_compile.triton('triton_poi_fused_convolution_91', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_91', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_91(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 196
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 100352*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/uz/cuzpktfgrhob2ahsuejeukwg535sstcozuqp32d5eytf3b52rtnw.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_50, triton_kernel_wrapper_mutation_51
# Graph fragment:
#   %triton_kernel_wrapper_mutation_51 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 395, constant_args_idx: 570, grid: [(512, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1091, SUM: %as_strided_default_53, SUMSQ: %as_strided_default_55, M: 25088, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_50 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 396, constant_args_idx: 571, grid: [(512, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1091, MEAN: %div_132, INVSTD: %rsqrt_44, GAMMA: %primals_268, BETA: %primals_269, Y: %permute_129, X_hat: %permute_130, M: 25088, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_92 = async_compile.triton('triton_poi_fused_92', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_92', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 25690112, 'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_92(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 25088*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 49*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 49*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_23 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_23', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/am/camgff55zj5fwf2tijs4aqz6xmwjsiwyu2o2n5ec3yibfcjjja5d.py
# Topologically Sorted Source Nodes: [layer4_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_50
#   layer4_0_bn2 => add_190, clamp_min_88, div_132, div_133, full_default_260, mul_309, rsqrt_44, sub_88
# Graph fragment:
#   %full_default_260 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 25088.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_132 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_53, %full_default_260), kwargs = {})
#   %div_133 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_55, %full_default_260), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_132, %div_132), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_133, %mul_309), kwargs = {})
#   %clamp_min_88 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_88, 0.0), kwargs = {})
#   %add_190 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_88, 1e-05), kwargs = {})
#   %rsqrt_44 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_190,), kwargs = {})
#   %triton_kernel_wrapper_mutation_50 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 396, constant_args_idx: 571, grid: [(512, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1091, MEAN: %div_132, INVSTD: %rsqrt_44, GAMMA: %primals_268, BETA: %primals_269, Y: %permute_129, X_hat: %permute_130, M: 25088, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_93 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_93', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_93', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_93(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 3.985969387755102e-05
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_24 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_24', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/m4/cm4fs57cmkgaonzaldiwivjnl4ivx6adi3d3tr7ds6e7ys3ybw5j.py
# Topologically Sorted Source Nodes: [layer4_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer4_0_relu_1 => full_default_262
# Graph fragment:
#   %full_default_262 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([512, 512, 7, 7], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_94 = async_compile.triton('triton_poi_fused_zeros_94', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_94', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25690112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_94(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wk/cwk6nxwliqt73w35kkae7bwxmzftlb3ssjw5ik6ceq4yxbwso3gp.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_25
# Graph fragment:
#   %clone_default_25 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_50,), kwargs = {})
triton_poi_fused_95 = async_compile.triton('triton_poi_fused_95', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_95', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 38535168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_95(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yj/cyj6t4poz5zqwjk6wf7lg3nuimdz56me3jt7vbbf2ecpldirnydm.py
# Topologically Sorted Source Nodes: [layer4_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer4_0_relu_1 => full_default_263
# Graph fragment:
#   %full_default_263 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([25088, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_96 = async_compile.triton('triton_poi_fused_zeros_96', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_96', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3211264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_96(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ob/cobihsirlddyebruqvf4csynkrzbgxzulnbhb753ly6dnomzgdt7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_24
# Graph fragment:
#   %clone_default_24 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_48,), kwargs = {})
triton_poi_fused_97 = async_compile.triton('triton_poi_fused_97', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_97', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4816896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_97(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5r/c5rvnwxhmy2gnw6peqnvaslikr3ajjxoy3dpytpb7g2obsd62aut.py
# Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer4_0_conv3 => convert_element_type_46
# Graph fragment:
#   %convert_element_type_46 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_272, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_98 = async_compile.triton('triton_poi_fused__to_copy_98', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_98', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_98(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/eo/ceojtksz7r64ceupjqiafvstqldvee76uruecorimnmfi4kvddri.py
# Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_conv3 => convolution_45
# Graph fragment:
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_131, %convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_99 = async_compile.triton('triton_poi_fused_convolution_99', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_99', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 51380224, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_99(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 25088*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/q6/cq6cbjf4nlk63ode3u5aw5p5hc72ko4mwglwkx7j6bh6ocy47jsg.py
# Topologically Sorted Source Nodes: [layer4_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer4_0_bn3 => full_default_264
# Graph fragment:
#   %full_default_264 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([2048], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_100 = async_compile.triton('triton_poi_fused_zeros_100', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_100', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_100(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/zu/czu44meb7hah77aexodjbc277j7p7v6zxxd4mkyvipsjylwo6273.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_20, clone_default_21, clone_default_22, clone_default_23
# Graph fragment:
#   %clone_default_22 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_44,), kwargs = {})
#   %clone_default_23 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_46,), kwargs = {})
#   %clone_default_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_40,), kwargs = {})
#   %clone_default_21 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_42,), kwargs = {})
triton_poi_fused_101 = async_compile.triton('triton_poi_fused_101', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_101', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 73728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_101(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
    tl.store(out_ptr3 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ef/cefaofm5ka2azo5hyztlp4vzhrmh5fsyx5gq342ex4hidmkvsvyo.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_44, triton_kernel_wrapper_mutation_45
# Graph fragment:
#   %triton_kernel_wrapper_mutation_45 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 399, constant_args_idx: 576, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1116, SUM: %as_strided_default_45, SUMSQ: %as_strided_default_47, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_44 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 400, constant_args_idx: 577, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1116, MEAN: %div_135, INVSTD: %rsqrt_45, GAMMA: %primals_274, BETA: %primals_275, Y: %permute_132, X_hat: %permute_133, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_102 = async_compile.triton('triton_poi_fused_102', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_102', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 102760448, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_102(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 2048*x2 + 100352*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 49*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 49*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:447
_bn_fwd_reduce_kernel_25 = async_compile.triton('_bn_fwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_reduce_kernel_25', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'SUM': '*fp32', 'SUMSQ': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)   # channel
    pid = tl.program_id(1) # block along M

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jg/cjg3aaf3n6ssoodiqb7n26jywwfqhzjd6rwerjcrasov6synhnhg.py
# Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_44
#   layer4_0_bn2 => full_default_260
#   layer4_0_bn3 => add_194, clamp_min_90, div_135, div_136, mul_316, rsqrt_45, sub_90
# Graph fragment:
#   %full_default_260 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 25088.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_135 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_45, %full_default_260), kwargs = {})
#   %div_136 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_47, %full_default_260), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_135, %div_135), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_136, %mul_316), kwargs = {})
#   %clamp_min_90 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_90, 0.0), kwargs = {})
#   %add_194 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_90, 1e-05), kwargs = {})
#   %rsqrt_45 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_194,), kwargs = {})
#   %triton_kernel_wrapper_mutation_44 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 400, constant_args_idx: 577, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1116, MEAN: %div_135, INVSTD: %rsqrt_45, GAMMA: %primals_274, BETA: %primals_275, Y: %permute_132, X_hat: %permute_133, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 3.985969387755102e-05
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp5 = tmp4 * tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:473
_bn_fwd_norm_kernel_26 = async_compile.triton('_bn_fwd_norm_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_fwd_norm_kernel_26', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'MEAN': '*fp32', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'BETA': '*fp32', 'Y': '*bf16', 'X_hat': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_fwd_norm_kernel(
    X, MEAN, INVSTD, GAMMA, BETA, Y, X_hat,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(0)
    pid = tl.program_id(1)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    y_ptrs = Y + n * stride_n + c * stride_c + s
    x_hat_ptrs = X_hat + n * stride_n + c * stride_c + s

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean   = tl.load(MEAN   + c).to(tl.float32)
    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)
    beta   = tl.load(BETA   + c).to(tl.float32)

    x_hat = (x - mean) * invstd
    # x_hat = tl.minimum(tl.maximum(x_hat, -5.0), 5.0)
    y = x_hat * gamma + beta

    tl.store(x_hat_ptrs, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/h5/ch5arpeexrcvmwl65bplivplc645alupfqevtmofrkwykxhuuecf.py
# Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer4_0_downsample_0 => convert_element_type_47
# Graph fragment:
#   %convert_element_type_47 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_278, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_104 = async_compile.triton('triton_poi_fused__to_copy_104', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_104', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_104(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/57/c57eq4gehlw2cyznwdv7au6nqbo6cax573ajzsike3be6vgm3ewr.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_38
# Graph fragment:
#   %triton_kernel_wrapper_mutation_38 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 583, grid: [(50176, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %add_201, Y_ptr: %permute_136, Mask_prt: %as_strided_default_39, n_elts: 51380224, BLOCK_SIZE: 1024}})
triton_poi_fused_105 = async_compile.triton('triton_poi_fused_105', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_105', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_105(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ph/cphrqjy63bi3wvwamx2heuwuxb7l57nd6xei65xzqppcac5mxiet.py
# Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_1_conv1 => convolution_47
# Graph fragment:
#   %convolution_47 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_136, %convert_element_type_48, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_106 = async_compile.triton('triton_poi_fused_convolution_106', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_106', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_106(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 2048*x2 + 100352*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/al/caldrlc6iqlsxnyxhlmhdkclp6xatzdxirxw26kqyzkmtq3arioo.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_11, clone_default_15
# Graph fragment:
#   %clone_default_15 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_30,), kwargs = {})
#   %clone_default_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_22,), kwargs = {})
triton_poi_fused_107 = async_compile.triton('triton_poi_fused_107', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_107', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64225280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_107(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xx/cxxerh23mj2fgvcdwx3wd7xu5uzd6cl3s3ub2n6dhha52ejlddqu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_10, clone_default_14
# Graph fragment:
#   %clone_default_14 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_28,), kwargs = {})
#   %clone_default_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_20,), kwargs = {})
triton_poi_fused_108 = async_compile.triton('triton_poi_fused_108', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'out_ptr1': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_108', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8028160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_108(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xs/cxsu5ch3jg52ryhf2pllhzatz7ykkhq5sbuksqa2aslgvsrbpcx5.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_8, clone_default_9
# Graph fragment:
#   %clone_default_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_16,), kwargs = {})
#   %clone_default_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_18,), kwargs = {})
triton_poi_fused_109 = async_compile.triton('triton_poi_fused_109', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_109', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 40960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_109(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/t3/ct3gpecc4odxzazlcghrzfabhofna7jtudbgtg2b645mbxpasctk.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_1, clone_default_4, clone_default_5
# Graph fragment:
#   %clone_default_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_8,), kwargs = {})
#   %clone_default_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_10,), kwargs = {})
#   %clone_default_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_2,), kwargs = {})
triton_poi_fused_110 = async_compile.triton('triton_poi_fused_110', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_110', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_110(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/am/cam5mitbrlohmhnpvjw4skex2ile7pxzyngy7wn6lsr6rg5lt3ud.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default
# Graph fragment:
#   %clone_default : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default,), kwargs = {})
triton_poi_fused_111 = async_compile.triton('triton_poi_fused_111', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_111', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_111(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ra/cralbhxc4cp4fcr6lubwqlhozkyouwd5erdxqmsb7a53fvkqijjp.py
# Topologically Sorted Source Nodes: [avgpool, flatten], Original ATen: [aten.mean, aten.view]
# Source node to ATen node mapping:
#   avgpool => mean
#   flatten => view_1303
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%permute_154, [-1, -2], True), kwargs = {})
#   %view_1303 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mean, [512, 2048]), kwargs = {})
triton_per_fused_mean_view_112 = async_compile.triton('triton_per_fused_mean_view_112', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_112', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_mean_view_112(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 49*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = 49.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/et/cetj22fnm5t3pcjo4o2yujizvboqojlp5feoajtsm3ti3axgf3go.py
# Topologically Sorted Source Nodes: [fc], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   fc => convert_element_type_54
# Graph fragment:
#   %convert_element_type_54 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_320, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_113 = async_compile.triton('triton_poi_fused__to_copy_113', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_113', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1638400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_113(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/nw/cnwcfzi5vj6ggzeoc3or2ksrba2yuesqu3cibcy5ioiwxxttm5ey.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, 1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_3, %add), kwargs = {})
triton_poi_fused_add_copy__114 = async_compile.triton('triton_poi_fused_add_copy__114', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__114', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__114(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ui/cuikg6t43sdjcqsmxtaa3q2cxluthzasmi66n2lldi3qstwggp26.py
# Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   bn1 => add_2, add_3, clamp_min, div, div_1, full_default_2, full_default_3, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, sub
# Graph fragment:
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 6422528.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_357, %full_default_2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_359, %full_default_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %div), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %mul_1), kwargs = {})
#   %clamp_min : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000001192092896), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min, %full_default_3), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, 0.9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 0.1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_7, 0.9), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, 0.1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_6, %add_2), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_7, %add_3), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_115 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_115', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_115', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_115(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 1.5570192920918366e-07
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000001192092896
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/66/c66hgyrk63cmxpabsnxep3bg2h2sxkzswsgd4p7ql7ybmtu5qfkt.py
# Topologically Sorted Source Nodes: [layer1_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer1_0_bn1 => add_6, add_7, clamp_min_2, div_3, div_4, full_default_8, full_default_9, mul_10, mul_11, mul_12, mul_13, mul_8, mul_9, sub_2
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_3 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_351, %full_default_8), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_353, %full_default_8), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %div_3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_4, %mul_8), kwargs = {})
#   %clamp_min_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %full_default_9 : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000005960464478), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_2, %full_default_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, 0.9), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, 0.1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_13, 0.9), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 0.1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %copy__4 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_12, %add_6), kwargs = {})
#   %copy__5 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_13, %add_7), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_116 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_116', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_116', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_116(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 6.228077168367346e-07
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000005960464478
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qh/cqhbiicdv77tdcfwfcno6eowlhdeeajq3carlyus4rud5hz6xhmc.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer1_0_bn1 => full_default_8, full_default_9
#   layer1_0_bn3 => add_14, add_15, clamp_min_6, div_10, div_9, mul_22, mul_23, mul_24, mul_25, mul_26, mul_27, sub_6
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_9 : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000005960464478), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_9 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_335, %full_default_8), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_337, %full_default_8), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %div_9), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_10, %mul_22), kwargs = {})
#   %clamp_min_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_6, 0.0), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_6, %full_default_9), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_24, 0.9), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, 0.1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %mul_25), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_25, 0.9), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, 0.1), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_27), kwargs = {})
#   %copy__10 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_24, %add_14), kwargs = {})
#   %copy__11 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_25, %add_15), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_117 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_117', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_117', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_117(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 6.228077168367346e-07
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000005960464478
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/et/cetahtzronuyd4ifuhpxh2v3uvtvnplq4em7tt4u3efkrhyvg7ch.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer2_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer1_0_bn1 => full_default_8, full_default_9
#   layer2_0_bn1 => add_49, add_50, clamp_min_22, div_33, div_34, mul_78, mul_79, mul_80, mul_81, mul_82, mul_83, sub_22
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_9 : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000005960464478), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_33 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_283, %full_default_8), kwargs = {})
#   %div_34 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_285, %full_default_8), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_33, %div_33), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_34, %mul_78), kwargs = {})
#   %clamp_min_22 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_22, 0.0), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_22, %full_default_9), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_72, 0.9), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_33, 0.1), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %mul_81), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_73, 0.9), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, 0.1), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, %mul_83), kwargs = {})
#   %copy__34 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_72, %add_49), kwargs = {})
#   %copy__35 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_73, %add_50), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_118 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_118', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_118', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_118(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 6.228077168367346e-07
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000005960464478
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/iv/civrogwxoik22nd2iuyqbgef23zqe2sk2rchfwgz2cvtyuof2bt5.py
# Topologically Sorted Source Nodes: [layer2_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer2_0_bn2 => add_53, add_54, clamp_min_24, div_36, div_37, full_default_72, full_default_73, mul_85, mul_86, mul_87, mul_88, mul_89, mul_90, sub_24
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_36 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_277, %full_default_72), kwargs = {})
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_279, %full_default_72), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_36, %div_36), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_37, %mul_85), kwargs = {})
#   %clamp_min_24 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0.0), kwargs = {})
#   %full_default_73 : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000025033950806), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_24, %full_default_73), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_78, 0.9), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_36, 0.1), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %mul_88), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_79, 0.9), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, 0.1), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %mul_90), kwargs = {})
#   %copy__37 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_78, %add_53), kwargs = {})
#   %copy__38 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_79, %add_54), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_119 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_119', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_119', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_119(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 2.4912308673469386e-06
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000025033950806
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ni/cniia26q4gzbai5auuyu2r4l5xvz2v4qlmlugutihetjhsgrgvdy.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer2_0_bn2 => full_default_72, full_default_73
#   layer2_0_bn3 => add_57, add_58, clamp_min_26, div_39, div_40, mul_92, mul_93, mul_94, mul_95, mul_96, mul_97, sub_26
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_73 : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000025033950806), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_39 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_269, %full_default_72), kwargs = {})
#   %div_40 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_271, %full_default_72), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_39, %div_39), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_40, %mul_92), kwargs = {})
#   %clamp_min_26 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_26, 0.0), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_26, %full_default_73), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_84, 0.9), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_39, 0.1), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %mul_95), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_85, 0.9), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, 0.1), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %mul_97), kwargs = {})
#   %copy__40 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_84, %add_57), kwargs = {})
#   %copy__41 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_85, %add_58), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_120 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_120', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_120', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_120(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 2.4912308673469386e-06
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000025033950806
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3i/c3ibel3j4dswvjp4elnpevbhigboxuq4bcybv7yxbp7fnop7uptx.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer3_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer2_0_bn2 => full_default_72, full_default_73
#   layer3_0_bn1 => add_105, add_106, clamp_min_48, div_72, div_73, mul_169, mul_170, mul_171, mul_172, mul_173, mul_174, sub_48
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_73 : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000025033950806), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_72 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_193, %full_default_72), kwargs = {})
#   %div_73 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_195, %full_default_72), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_72, %div_72), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_73, %mul_169), kwargs = {})
#   %clamp_min_48 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_48, 0.0), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_48, %full_default_73), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_150, 0.9), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_72, 0.1), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_171, %mul_172), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_151, 0.9), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_170, 0.1), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %mul_174), kwargs = {})
#   %copy__73 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_150, %add_105), kwargs = {})
#   %copy__74 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_151, %add_106), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_121 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_121', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_121', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_121(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 2.4912308673469386e-06
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000025033950806
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/si/csi3gv4h2ke2h6o2ykpm5bsium2cbgv3kjubzl7dmfvyk7asbwfn.py
# Topologically Sorted Source Nodes: [layer3_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer3_0_bn2 => add_109, add_110, clamp_min_50, div_75, div_76, full_default_148, full_default_149, mul_176, mul_177, mul_178, mul_179, mul_180, mul_181, sub_50
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_75 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_187, %full_default_148), kwargs = {})
#   %div_76 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_189, %full_default_148), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_75, %div_75), kwargs = {})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_76, %mul_176), kwargs = {})
#   %clamp_min_50 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_50, 0.0), kwargs = {})
#   %full_default_149 : [num_users=19] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000100135803223), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_177 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_50, %full_default_149), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_156, 0.9), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_75, 0.1), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_178, %mul_179), kwargs = {})
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_157, 0.9), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_177, 0.1), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_180, %mul_181), kwargs = {})
#   %copy__76 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_156, %add_109), kwargs = {})
#   %copy__77 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_157, %add_110), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_122 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_122', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_122', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_122(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 9.964923469387754e-06
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000100135803223
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/n5/cn5psllxvzzjkcbir4lljzkzhudtd3zywjo3bfwrkg3obv77lotr.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer3_0_bn2 => full_default_148, full_default_149
#   layer3_0_bn3 => add_113, add_114, clamp_min_52, div_78, div_79, mul_183, mul_184, mul_185, mul_186, mul_187, mul_188, sub_52
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_149 : [num_users=19] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000100135803223), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_78 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_179, %full_default_148), kwargs = {})
#   %div_79 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_181, %full_default_148), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_78, %div_78), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_79, %mul_183), kwargs = {})
#   %clamp_min_52 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_52, 0.0), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_52, %full_default_149), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_162, 0.9), kwargs = {})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_78, 0.1), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %mul_186), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_163, 0.9), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, 0.1), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_187, %mul_188), kwargs = {})
#   %copy__79 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_162, %add_113), kwargs = {})
#   %copy__80 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_163, %add_114), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_123 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_123', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_123', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_123(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 9.964923469387754e-06
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000100135803223
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mj/cmjucmedgojom5gird4wweyz4ba6skx6c4htt4bjxp5u7vnoylcp.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer4_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer3_0_bn2 => full_default_148, full_default_149
#   layer4_0_bn1 => add_187, add_188, clamp_min_86, div_129, div_130, mul_302, mul_303, mul_304, mul_305, mul_306, mul_307, sub_86
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_149 : [num_users=19] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000100135803223), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_129 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_59, %full_default_148), kwargs = {})
#   %div_130 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_61, %full_default_148), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_129, %div_129), kwargs = {})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_130, %mul_302), kwargs = {})
#   %clamp_min_86 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_86, 0.0), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_86, %full_default_149), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_264, 0.9), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_129, 0.1), kwargs = {})
#   %add_187 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_304, %mul_305), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_265, 0.9), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_303, 0.1), kwargs = {})
#   %add_188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_306, %mul_307), kwargs = {})
#   %copy__130 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_264, %add_187), kwargs = {})
#   %copy__131 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_265, %add_188), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_124 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_124', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_124', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_124(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 9.964923469387754e-06
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.0000100135803223
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/k2/ck2p6mxhn6biuodh4v6q3ia2yuuxeossnrixvy3d3imug3n37okj.py
# Topologically Sorted Source Nodes: [layer4_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer4_0_bn2 => add_191, add_192, clamp_min_88, div_132, div_133, full_default_260, full_default_261, mul_309, mul_310, mul_311, mul_312, mul_313, mul_314, sub_88
# Graph fragment:
#   %full_default_260 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 25088.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_132 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_53, %full_default_260), kwargs = {})
#   %div_133 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_55, %full_default_260), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_132, %div_132), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_133, %mul_309), kwargs = {})
#   %clamp_min_88 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_88, 0.0), kwargs = {})
#   %full_default_261 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 1.00003981590271), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_88, %full_default_261), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_270, 0.9), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_132, 0.1), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %mul_312), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_271, 0.9), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, 0.1), kwargs = {})
#   %add_192 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_313, %mul_314), kwargs = {})
#   %copy__133 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_270, %add_191), kwargs = {})
#   %copy__134 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_271, %add_192), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_125 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_125', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_125', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_125(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 3.985969387755102e-05
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.00003981590271
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qa/cqaiprujn4rcgkkfrttym6q422btkwu3pzzxoujxn3cncepphstx.py
# Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer4_0_bn2 => full_default_260, full_default_261
#   layer4_0_bn3 => add_195, add_196, clamp_min_90, div_135, div_136, mul_316, mul_317, mul_318, mul_319, mul_320, mul_321, sub_90
# Graph fragment:
#   %full_default_260 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 25088.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_261 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 1.00003981590271), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_135 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_45, %full_default_260), kwargs = {})
#   %div_136 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_47, %full_default_260), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_135, %div_135), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_136, %mul_316), kwargs = {})
#   %clamp_min_90 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_90, 0.0), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_90, %full_default_261), kwargs = {})
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_276, 0.9), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_135, 0.1), kwargs = {})
#   %add_195 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_318, %mul_319), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_277, 0.9), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_317, 0.1), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_320, %mul_321), kwargs = {})
#   %copy__136 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_276, %add_195), kwargs = {})
#   %copy__137 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_277, %add_196), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_126 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_126', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_126', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_126(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask)
    tmp1 = 0.9
    tmp2 = tmp0 * tmp1
    tmp4 = 3.985969387755102e-05
    tmp5 = tmp3 * tmp4
    tmp6 = 0.1
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tmp11 * tmp4
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 - tmp13
    tmp15 = 0.0
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = 1.00003981590271
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp6
    tmp20 = tmp10 + tmp19
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp20, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (512, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(primals_3, (), ())
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_9, (), ())
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_15, (), ())
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_21, (), ())
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (256, ), (1, ))
    assert_size_stride(primals_26, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_27, (), ())
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_32, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_33, (), ())
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_39, (), ())
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_45, (), ())
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (256, ), (1, ))
    assert_size_stride(primals_50, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_51, (), ())
    assert_size_stride(primals_52, (64, ), (1, ))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_57, (), ())
    assert_size_stride(primals_58, (64, ), (1, ))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (64, ), (1, ))
    assert_size_stride(primals_62, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_63, (), ())
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_69, (), ())
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_75, (), ())
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_81, (), ())
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_86, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_87, (), ())
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_93, (), ())
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_98, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_99, (), ())
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (128, ), (1, ))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_105, (), ())
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_108, (512, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_110, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_111, (), ())
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_117, (), ())
    assert_size_stride(primals_118, (128, ), (1, ))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_123, (), ())
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_129, (), ())
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_135, (), ())
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_141, (), ())
    assert_size_stride(primals_142, (512, ), (1, ))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, ), (1, ))
    assert_size_stride(primals_146, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_147, (), ())
    assert_size_stride(primals_148, (256, ), (1, ))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_151, (256, ), (1, ))
    assert_size_stride(primals_152, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_153, (), ())
    assert_size_stride(primals_154, (256, ), (1, ))
    assert_size_stride(primals_155, (256, ), (1, ))
    assert_size_stride(primals_156, (256, ), (1, ))
    assert_size_stride(primals_157, (256, ), (1, ))
    assert_size_stride(primals_158, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_159, (), ())
    assert_size_stride(primals_160, (1024, ), (1, ))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_163, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_165, (), ())
    assert_size_stride(primals_166, (1024, ), (1, ))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_168, (1024, ), (1, ))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_170, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_171, (), ())
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (256, ), (1, ))
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_177, (), ())
    assert_size_stride(primals_178, (256, ), (1, ))
    assert_size_stride(primals_179, (256, ), (1, ))
    assert_size_stride(primals_180, (256, ), (1, ))
    assert_size_stride(primals_181, (256, ), (1, ))
    assert_size_stride(primals_182, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_183, (), ())
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (1024, ), (1, ))
    assert_size_stride(primals_187, (1024, ), (1, ))
    assert_size_stride(primals_188, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_189, (), ())
    assert_size_stride(primals_190, (256, ), (1, ))
    assert_size_stride(primals_191, (256, ), (1, ))
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_194, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_195, (), ())
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_197, (256, ), (1, ))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, ), (1, ))
    assert_size_stride(primals_200, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_201, (), ())
    assert_size_stride(primals_202, (1024, ), (1, ))
    assert_size_stride(primals_203, (1024, ), (1, ))
    assert_size_stride(primals_204, (1024, ), (1, ))
    assert_size_stride(primals_205, (1024, ), (1, ))
    assert_size_stride(primals_206, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_207, (), ())
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_209, (256, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_212, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_213, (), ())
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_215, (256, ), (1, ))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (256, ), (1, ))
    assert_size_stride(primals_218, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (1024, ), (1, ))
    assert_size_stride(primals_221, (1024, ), (1, ))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_223, (1024, ), (1, ))
    assert_size_stride(primals_224, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_227, (256, ), (1, ))
    assert_size_stride(primals_228, (256, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_230, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (256, ), (1, ))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_239, (1024, ), (1, ))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_242, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_245, (256, ), (1, ))
    assert_size_stride(primals_246, (256, ), (1, ))
    assert_size_stride(primals_247, (256, ), (1, ))
    assert_size_stride(primals_248, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_252, (256, ), (1, ))
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_254, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (1024, ), (1, ))
    assert_size_stride(primals_257, (1024, ), (1, ))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_259, (1024, ), (1, ))
    assert_size_stride(primals_260, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (512, ), (1, ))
    assert_size_stride(primals_263, (512, ), (1, ))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (512, ), (1, ))
    assert_size_stride(primals_266, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (512, ), (1, ))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (512, ), (1, ))
    assert_size_stride(primals_272, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (2048, ), (1, ))
    assert_size_stride(primals_275, (2048, ), (1, ))
    assert_size_stride(primals_276, (2048, ), (1, ))
    assert_size_stride(primals_277, (2048, ), (1, ))
    assert_size_stride(primals_278, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (2048, ), (1, ))
    assert_size_stride(primals_281, (2048, ), (1, ))
    assert_size_stride(primals_282, (2048, ), (1, ))
    assert_size_stride(primals_283, (2048, ), (1, ))
    assert_size_stride(primals_284, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (512, ), (1, ))
    assert_size_stride(primals_290, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (2048, ), (1, ))
    assert_size_stride(primals_299, (2048, ), (1, ))
    assert_size_stride(primals_300, (2048, ), (1, ))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_302, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (512, ), (1, ))
    assert_size_stride(primals_308, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, ), (1, ))
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (512, ), (1, ))
    assert_size_stride(primals_314, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (2048, ), (1, ))
    assert_size_stride(primals_317, (2048, ), (1, ))
    assert_size_stride(primals_318, (2048, ), (1, ))
    assert_size_stride(primals_319, (2048, ), (1, ))
    assert_size_stride(primals_320, (100, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((150528, 32), (32, 1), torch.int32)
        buf1 = empty_strided_cuda((150528, ), (1, ), torch.bfloat16)
        buf2 = empty_strided_cuda((150528, ), (1, ), torch.bfloat16)
        buf3 = empty_strided_cuda((150528, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_2, buf3, 77070336, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(buf3, buf0, buf1, buf2, 512, 1, 32, 1, 2, 16, 32, 3, 150528, 1, 1, stream=stream0)
        buf7 = reinterpret_tensor(buf3, (512, 3, 224, 224), (150528, 1, 672, 3), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_2, buf7, 1536, 50176, stream=stream0)
        del primals_2
        buf8 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.bfloat16)
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(primals_1, buf8, 192, 49, stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (512, 64, 112, 112), (802816, 1, 7168, 64), 'torch.ops.aten.convolution.default')
        del buf7
        del buf8
        buf10 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_3.run(buf10, 64, stream=stream0)
        buf11 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf12 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf49 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf50 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf84 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf85 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(buf10, buf11, buf12, buf49, buf50, buf84, buf85, 64, stream=stream0)
        buf13 = empty_strided_cuda((512, 64, 12544), (802816, 12544, 1), torch.bfloat16)
        buf19 = empty_strided_cuda((512, 64, 12544), (802816, 12544, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(buf9, buf13, buf19, 32768, 12544, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_1.run(buf13, buf11, buf12, 6422528, 12544, 802816, 12544, 1024, 64, 6272, 1, stream=stream0)
        buf16 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf20 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_6.run(buf12, buf11, buf16, buf20, 64, stream=stream0)
        buf17 = buf13; del buf13  # reuse
        buf18 = reinterpret_tensor(buf9, (512, 64, 12544), (802816, 12544, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_2.run(buf19, buf20, buf16, primals_4, primals_5, buf17, buf18, 6422528, 12544, 802816, 12544, 1024, 64, 6272, 1, stream=stream0)
        del primals_5
        buf23 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf24 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf25 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf18, (802816, 512), (512, 1), 0), buf23, buf24, buf25, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf29 = reinterpret_tensor(buf18, (512, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf18  # reuse
        buf31 = reinterpret_tensor(buf19, (512, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [relu, ], Original ATen: [aten.empty_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_empty_like_7.run(buf29, buf31, 411041792, stream=stream0)
        buf32 = empty_strided_cuda((512, 64, 112, 112), (802816, 12544, 112, 1), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(buf32, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf17, (512, 64, 112, 112), (802816, 12544, 112, 1), 0), buf31, buf32, 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf35 = empty_strided_cuda((802816, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_9.run(buf35, 12845056, stream=stream0)
        buf36 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(buf35, buf36, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf32, (802816, 512), (512, 1), 0), reinterpret_tensor(buf36, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf38 = empty_strided_cuda((512, 64, 56, 56), (200704, 3136, 56, 1), torch.bfloat16)
        buf39 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.int8)
        buf47 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        buf143 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [maxpool, layer1_0_conv1, layer1_0_downsample_0], Original ATen: [aten.max_pool2d_with_indices, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_11.run(buf31, buf38, buf39, buf47, buf143, 32768, 3136, stream=stream0)
        buf40 = empty_strided_cuda((64, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(primals_8, buf40, 4096, stream=stream0)
        del primals_8
        buf41 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf42 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf43 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf38, (200704, 512), (512, 1), 0), buf41, buf42, buf43, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, buf40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf51 = reinterpret_tensor(buf47, (512, 64, 3136), (200704, 3136, 1), 0); del buf47  # reuse
        buf57 = empty_strided_cuda((512, 64, 3136), (200704, 3136, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(buf48, buf51, buf57, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf51, buf49, buf50, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf54 = buf20; del buf20  # reuse
        buf58 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14.run(buf50, buf49, buf54, buf58, 64, stream=stream0)
        buf55 = buf51; del buf51  # reuse
        buf56 = reinterpret_tensor(buf48, (512, 64, 3136), (200704, 3136, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf57, buf58, buf54, primals_10, primals_11, buf55, buf56, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del buf57
        del primals_11
        buf61 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf62 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf63 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf56, (200704, 512), (512, 1), 0), buf61, buf62, buf63, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf67 = empty_strided_cuda((512, 64, 56, 56), (200704, 3136, 56, 1), torch.int8)
        # Topologically Sorted Source Nodes: [layer1_0_relu], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_15.run(buf67, 102760448, stream=stream0)
        buf68 = reinterpret_tensor(buf56, (512, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf56  # reuse
        buf69 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf103 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(buf67, buf69, buf103, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf55, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), buf68, reinterpret_tensor(buf69, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf72 = empty_strided_cuda((200704, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer1_0_relu], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_17.run(buf72, 3211264, stream=stream0)
        buf73 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        buf106 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(buf72, buf73, buf106, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf69, (200704, 512), (512, 1), 0), reinterpret_tensor(buf73, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf75 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_19.run(primals_14, buf75, 4096, 9, stream=stream0)
        del primals_14
        buf76 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf77 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf78 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf68, (200704, 512), (512, 1), 0), buf76, buf77, buf78, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf82 = reinterpret_tensor(buf55, (512, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf68, buf82, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, buf75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf86 = reinterpret_tensor(buf82, (512, 64, 3136), (200704, 3136, 1), 0); del buf82  # reuse
        buf92 = reinterpret_tensor(buf68, (512, 64, 3136), (200704, 3136, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(buf83, buf86, buf92, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf86, buf84, buf85, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf89 = buf58; del buf58  # reuse
        buf93 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14.run(buf85, buf84, buf89, buf93, 64, stream=stream0)
        buf90 = buf86; del buf86  # reuse
        buf91 = reinterpret_tensor(buf83, (512, 64, 3136), (200704, 3136, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf92, buf93, buf89, primals_16, primals_17, buf90, buf91, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del buf92
        del primals_17
        buf96 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf97 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf98 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf91, (200704, 512), (512, 1), 0), buf96, buf97, buf98, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf102 = reinterpret_tensor(buf91, (512, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf90, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), buf102, reinterpret_tensor(buf103, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf103, (200704, 512), (512, 1), 0), reinterpret_tensor(buf106, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf108 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(primals_20, buf108, 16384, stream=stream0)
        del primals_20
        buf109 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf110 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf111 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf102, (200704, 512), (512, 1), 0), buf109, buf110, buf111, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf115 = reinterpret_tensor(buf90, (512, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf102, buf115, 32768, 3136, stream=stream0)
        del buf102
        # Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, buf108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        del buf115
        buf117 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_22.run(buf117, 256, stream=stream0)
        buf118 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf119 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf145 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf146 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_23.run(buf117, buf118, buf119, buf145, buf146, 256, stream=stream0)
        buf120 = reinterpret_tensor(buf17, (512, 256, 3136), (802816, 3136, 1), 0); del buf17  # reuse
        buf126 = reinterpret_tensor(buf29, (512, 256, 3136), (802816, 3136, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(buf116, buf120, buf126, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_7.run(buf120, buf118, buf119, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf123 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf127 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_25.run(buf119, buf118, buf123, buf127, 256, stream=stream0)
        buf124 = buf120; del buf120  # reuse
        buf125 = reinterpret_tensor(buf116, (512, 256, 3136), (802816, 3136, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_8.run(buf126, buf127, buf123, primals_22, primals_23, buf124, buf125, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_23
        buf130 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf131 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf132 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf125, (802816, 512), (512, 1), 0), buf130, buf131, buf132, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf136 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_downsample_0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(primals_26, buf136, 16384, stream=stream0)
        del primals_26
        buf137 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf138 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf139 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf38, (200704, 512), (512, 1), 0), buf137, buf138, buf139, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_downsample_0], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, buf136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf147 = buf125; del buf125  # reuse
        buf153 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(buf144, buf147, buf153, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_7.run(buf147, buf145, buf146, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf150 = buf127; del buf127  # reuse
        buf154 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_downsample_1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_25.run(buf146, buf145, buf150, buf154, 256, stream=stream0)
        buf151 = buf147; del buf147  # reuse
        buf152 = reinterpret_tensor(buf144, (512, 256, 3136), (802816, 3136, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_8.run(buf153, buf154, buf150, primals_28, primals_29, buf151, buf152, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_29
        buf157 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf158 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf159 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf152, (802816, 512), (512, 1), 0), buf157, buf158, buf159, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf163 = reinterpret_tensor(buf32, (512, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_relu_2], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(buf163, 411041792, stream=stream0)
        buf164 = reinterpret_tensor(buf152, (512, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf152  # reuse
        buf165 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf163, buf165, 411041792, stream=stream0)
        buf166 = reinterpret_tensor(buf153, (512, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_27.run(buf124, buf151, buf166, 411041792, stream=stream0)
        del buf124
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf166, buf164, reinterpret_tensor(buf165, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf169 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(buf35, buf169, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf165, (802816, 512), (512, 1), 0), reinterpret_tensor(buf169, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf171 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(primals_32, buf171, 16384, stream=stream0)
        del primals_32
        buf172 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf173 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf174 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf164, (802816, 512), (512, 1), 0), buf172, buf173, buf174, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf178 = reinterpret_tensor(buf166, (512, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf164, buf178, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, buf171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf180 = buf93; del buf93  # reuse
        buf181 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf213 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf214 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf10, buf180, buf181, buf213, buf214, 64, stream=stream0)
        buf182 = reinterpret_tensor(buf143, (512, 64, 3136), (200704, 3136, 1), 0); del buf143  # reuse
        buf188 = reinterpret_tensor(buf38, (512, 64, 3136), (200704, 3136, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(buf179, buf182, buf188, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf182, buf180, buf181, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf185 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf189 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14.run(buf181, buf180, buf185, buf189, 64, stream=stream0)
        buf186 = buf182; del buf182  # reuse
        buf187 = reinterpret_tensor(buf179, (512, 64, 3136), (200704, 3136, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf188, buf189, buf185, primals_34, primals_35, buf186, buf187, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del buf188
        del primals_35
        buf192 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf193 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf194 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf187, (200704, 512), (512, 1), 0), buf192, buf193, buf194, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf198 = reinterpret_tensor(buf187, (512, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf187  # reuse
        buf199 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf232 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(buf67, buf199, buf232, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf186, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), buf198, reinterpret_tensor(buf199, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf202 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        buf235 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(buf72, buf202, buf235, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf199, (200704, 512), (512, 1), 0), reinterpret_tensor(buf202, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf204 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_1_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_19.run(primals_38, buf204, 4096, 9, stream=stream0)
        del primals_38
        buf205 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf206 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf207 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf198, (200704, 512), (512, 1), 0), buf205, buf206, buf207, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf211 = reinterpret_tensor(buf186, (512, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf198, buf211, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_1_conv2], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, buf204, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf215 = reinterpret_tensor(buf211, (512, 64, 3136), (200704, 3136, 1), 0); del buf211  # reuse
        buf221 = reinterpret_tensor(buf198, (512, 64, 3136), (200704, 3136, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(buf212, buf215, buf221, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf215, buf213, buf214, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf218 = buf189; del buf189  # reuse
        buf222 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14.run(buf214, buf213, buf218, buf222, 64, stream=stream0)
        buf219 = buf215; del buf215  # reuse
        buf220 = reinterpret_tensor(buf212, (512, 64, 3136), (200704, 3136, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf221, buf222, buf218, primals_40, primals_41, buf219, buf220, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del buf221
        del primals_41
        buf225 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf226 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf227 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf220, (200704, 512), (512, 1), 0), buf225, buf226, buf227, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf231 = reinterpret_tensor(buf220, (512, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf219, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), buf231, reinterpret_tensor(buf232, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf232, (200704, 512), (512, 1), 0), reinterpret_tensor(buf235, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf237 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_1_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(primals_44, buf237, 16384, stream=stream0)
        del primals_44
        buf238 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf239 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf240 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf231, (200704, 512), (512, 1), 0), buf238, buf239, buf240, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf244 = reinterpret_tensor(buf219, (512, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf231, buf244, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_1_conv3], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, buf237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf246 = buf154; del buf154  # reuse
        buf247 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf117, buf246, buf247, 256, stream=stream0)
        buf248 = reinterpret_tensor(buf178, (512, 256, 3136), (802816, 3136, 1), 0); del buf178  # reuse
        buf254 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(buf245, buf248, buf254, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_7.run(buf248, buf246, buf247, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf251 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf255 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_25.run(buf247, buf246, buf251, buf255, 256, stream=stream0)
        buf252 = buf248; del buf248  # reuse
        buf253 = reinterpret_tensor(buf245, (512, 256, 3136), (802816, 3136, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_8.run(buf254, buf255, buf251, primals_46, primals_47, buf252, buf253, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_47
        buf258 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf259 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf260 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf253, (802816, 512), (512, 1), 0), buf258, buf259, buf260, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf264 = reinterpret_tensor(buf253, (512, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf253  # reuse
        buf265 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf163, buf265, 411041792, stream=stream0)
        buf266 = reinterpret_tensor(buf254, (512, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_27.run(buf252, buf164, buf266, 411041792, stream=stream0)
        del buf164
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf266, buf264, reinterpret_tensor(buf265, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf269 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(buf35, buf269, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf265, (802816, 512), (512, 1), 0), reinterpret_tensor(buf269, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf271 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(primals_50, buf271, 16384, stream=stream0)
        del primals_50
        buf272 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf273 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf274 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf264, (802816, 512), (512, 1), 0), buf272, buf273, buf274, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf278 = reinterpret_tensor(buf266, (512, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf264, buf278, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, buf271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf280 = buf222; del buf222  # reuse
        buf281 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf313 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_31.run(buf10, buf280, buf281, buf313, 64, stream=stream0)
        buf282 = reinterpret_tensor(buf244, (512, 64, 3136), (200704, 3136, 1), 0); del buf244  # reuse
        buf288 = reinterpret_tensor(buf231, (512, 64, 3136), (200704, 3136, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(buf279, buf282, buf288, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf282, buf280, buf281, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf285 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf289 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14.run(buf281, buf280, buf285, buf289, 64, stream=stream0)
        buf286 = buf282; del buf282  # reuse
        buf287 = reinterpret_tensor(buf279, (512, 64, 3136), (200704, 3136, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf288, buf289, buf285, primals_52, primals_53, buf286, buf287, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del buf288
        del primals_53
        buf292 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf293 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf294 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf287, (200704, 512), (512, 1), 0), buf292, buf293, buf294, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf298 = reinterpret_tensor(buf287, (512, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf287  # reuse
        buf299 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf67, buf299, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf286, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), buf298, reinterpret_tensor(buf299, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf302 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        buf333 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(buf72, buf302, buf333, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf299, (200704, 512), (512, 1), 0), reinterpret_tensor(buf302, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf304 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_2_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_19.run(primals_56, buf304, 4096, 9, stream=stream0)
        del primals_56
        buf305 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf306 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf307 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf298, (200704, 512), (512, 1), 0), buf305, buf306, buf307, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf311 = reinterpret_tensor(buf286, (512, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf298, buf311, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_2_conv2], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, buf304, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf314 = reinterpret_tensor(buf311, (512, 64, 3136), (200704, 3136, 1), 0); del buf311  # reuse
        buf320 = reinterpret_tensor(buf298, (512, 64, 3136), (200704, 3136, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(buf312, buf314, buf320, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf314, buf10, buf313, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf317 = buf289; del buf289  # reuse
        buf321 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_14.run(buf313, buf10, buf317, buf321, 64, stream=stream0)
        buf318 = buf314; del buf314  # reuse
        buf319 = reinterpret_tensor(buf312, (512, 64, 3136), (200704, 3136, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf320, buf321, buf317, primals_58, primals_59, buf318, buf319, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del buf320
        del buf321
        del primals_59
        buf324 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf325 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf326 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf319, (200704, 512), (512, 1), 0), buf324, buf325, buf326, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf330 = reinterpret_tensor(buf319, (512, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf318, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), buf330, buf67, 102760448, 1024, 100352, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf67, (200704, 512), (512, 1), 0), reinterpret_tensor(buf333, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf335 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_2_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(primals_62, buf335, 16384, stream=stream0)
        del primals_62
        buf336 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf337 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf338 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf330, (200704, 512), (512, 1), 0), buf336, buf337, buf338, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf342 = reinterpret_tensor(buf318, (512, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf330, buf342, 32768, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_2_conv3], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, buf335, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf344 = buf255; del buf255  # reuse
        buf345 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf117, buf344, buf345, 256, stream=stream0)
        buf346 = reinterpret_tensor(buf278, (512, 256, 3136), (802816, 3136, 1), 0); del buf278  # reuse
        buf352 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(buf343, buf346, buf352, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_7.run(buf346, buf344, buf345, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf349 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf353 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_25.run(buf345, buf344, buf349, buf353, 256, stream=stream0)
        buf350 = buf346; del buf346  # reuse
        buf351 = reinterpret_tensor(buf343, (512, 256, 3136), (802816, 3136, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_8.run(buf352, buf353, buf349, primals_64, primals_65, buf350, buf351, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_65
        buf356 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf357 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf358 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf351, (802816, 512), (512, 1), 0), buf356, buf357, buf358, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf362 = reinterpret_tensor(buf351, (512, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf351  # reuse
        buf363 = reinterpret_tensor(buf352, (512, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_27.run(buf350, buf264, buf363, 411041792, stream=stream0)
        del buf264
        del buf350
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf363, buf362, buf163, 411041792, 1024, 401408, 1, 1, stream=stream0)
        del buf165
        del buf265
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf163, (802816, 512), (512, 1), 0), buf35, 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del buf163
        buf367 = empty_strided_cuda((128, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(primals_68, buf367, 32768, stream=stream0)
        del primals_68
        buf368 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf369 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf370 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf362, (802816, 512), (512, 1), 0), buf368, buf369, buf370, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf374 = reinterpret_tensor(buf363, (512, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf362, buf374, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf374, buf367, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (512, 128, 56, 56), (401408, 1, 7168, 128), 'torch.ops.aten.convolution.default')
        buf376 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_34.run(buf376, 128, stream=stream0)
        buf377 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf378 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf411 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf412 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf376, buf377, buf378, buf411, buf412, 128, stream=stream0)
        buf379 = empty_strided_cuda((512, 128, 3136), (401408, 3136, 1), torch.bfloat16)
        buf385 = empty_strided_cuda((512, 128, 3136), (401408, 3136, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_36.run(buf375, buf379, buf385, 65536, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_9.run(buf379, buf377, buf378, 1605632, 3136, 401408, 3136, 1024, 128, 1568, 1, stream=stream0)
        buf382 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf386 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer2_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_37.run(buf378, buf377, buf382, buf386, 128, stream=stream0)
        buf383 = buf379; del buf379  # reuse
        buf384 = reinterpret_tensor(buf375, (512, 128, 3136), (401408, 3136, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_10.run(buf385, buf386, buf382, primals_70, primals_71, buf383, buf384, 1605632, 3136, 401408, 3136, 1024, 128, 1568, 1, stream=stream0)
        del buf385
        del primals_71
        buf389 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf390 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf391 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf384, (401408, 512), (512, 1), 0), buf389, buf390, buf391, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf395 = reinterpret_tensor(buf384, (512, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf384  # reuse
        buf396 = empty_strided_cuda((512, 128, 56, 56), (401408, 3136, 56, 1), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf396, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf383, (512, 128, 56, 56), (401408, 3136, 56, 1), 0), buf395, buf396, 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf399 = empty_strided_cuda((401408, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer2_0_relu], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_39.run(buf399, 6422528, stream=stream0)
        buf400 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_40.run(buf399, buf400, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf396, (401408, 512), (512, 1), 0), reinterpret_tensor(buf400, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf402 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_41.run(primals_74, buf402, 16384, 9, stream=stream0)
        del primals_74
        buf403 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf404 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf405 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf395, (401408, 512), (512, 1), 0), buf403, buf404, buf405, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf409 = reinterpret_tensor(buf383, (512, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf395, buf409, 65536, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, buf402, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf413 = empty_strided_cuda((512, 128, 784), (100352, 784, 1), torch.bfloat16)
        buf419 = empty_strided_cuda((512, 128, 784), (100352, 784, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf410, buf413, buf419, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf413, buf411, buf412, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf416 = buf386; del buf386  # reuse
        buf420 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44.run(buf412, buf411, buf416, buf420, 128, stream=stream0)
        buf417 = buf413; del buf413  # reuse
        buf418 = reinterpret_tensor(buf410, (512, 128, 784), (100352, 784, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf419, buf420, buf416, primals_76, primals_77, buf417, buf418, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del buf419
        del primals_77
        buf423 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf424 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf425 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf418, (100352, 512), (512, 1), 0), buf423, buf424, buf425, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf429 = empty_strided_cuda((512, 128, 28, 28), (100352, 784, 28, 1), torch.int8)
        # Topologically Sorted Source Nodes: [layer2_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_45.run(buf429, 51380224, stream=stream0)
        buf430 = reinterpret_tensor(buf418, (512, 128, 28, 28), (100352, 784, 28, 1), 0); del buf418  # reuse
        buf431 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_46.run(buf429, buf431, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf417, (512, 128, 28, 28), (100352, 784, 28, 1), 0), buf430, reinterpret_tensor(buf431, (512, 128, 28, 28), (100352, 784, 28, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf434 = empty_strided_cuda((100352, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer2_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_47.run(buf434, 1605632, stream=stream0)
        buf435 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf434, buf435, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf431, (100352, 512), (512, 1), 0), reinterpret_tensor(buf435, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf437 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(primals_80, buf437, 65536, stream=stream0)
        del primals_80
        buf438 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf439 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf440 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf430, (100352, 512), (512, 1), 0), buf438, buf439, buf440, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf444 = reinterpret_tensor(buf417, (512, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf430, buf444, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, buf437, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf446 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_51.run(buf446, 512, stream=stream0)
        buf447 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf448 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf474 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf475 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_52.run(buf446, buf447, buf448, buf474, buf475, 512, stream=stream0)
        buf449 = reinterpret_tensor(buf409, (512, 512, 784), (401408, 784, 1), 0); del buf409  # reuse
        buf455 = reinterpret_tensor(buf395, (512, 512, 784), (401408, 784, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_53.run(buf445, buf449, buf455, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf449, buf447, buf448, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf452 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf456 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_54.run(buf448, buf447, buf452, buf456, 512, stream=stream0)
        buf453 = buf449; del buf449  # reuse
        buf454 = reinterpret_tensor(buf445, (512, 512, 784), (401408, 784, 1), 0); del buf445  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf455, buf456, buf452, primals_82, primals_83, buf453, buf454, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_83
        buf459 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf460 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf461 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf454, (401408, 512), (512, 1), 0), buf459, buf460, buf461, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf465 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_55.run(primals_86, buf465, 131072, stream=stream0)
        del primals_86
        buf466 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf467 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf468 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf362, (802816, 512), (512, 1), 0), buf466, buf467, buf468, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf472 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_28.run(buf362, buf472, 131072, 3136, stream=stream0)
        del buf362
        # Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, buf465, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        del buf472
        buf476 = buf454; del buf454  # reuse
        buf482 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_53.run(buf473, buf476, buf482, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf476, buf474, buf475, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf479 = buf456; del buf456  # reuse
        buf483 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_downsample_1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_54.run(buf475, buf474, buf479, buf483, 512, stream=stream0)
        buf480 = buf476; del buf476  # reuse
        buf481 = reinterpret_tensor(buf473, (512, 512, 784), (401408, 784, 1), 0); del buf473  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf482, buf483, buf479, primals_88, primals_89, buf480, buf481, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_89
        buf486 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf487 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf488 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf481, (401408, 512), (512, 1), 0), buf486, buf487, buf488, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf492 = reinterpret_tensor(buf396, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_relu_2], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf492, 205520896, stream=stream0)
        buf493 = reinterpret_tensor(buf481, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf481  # reuse
        buf494 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_56.run(buf492, buf494, 205520896, stream=stream0)
        buf495 = reinterpret_tensor(buf482, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf482  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf453, buf480, buf495, 205520896, stream=stream0)
        del buf453
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf495, buf493, reinterpret_tensor(buf494, (512, 512, 28, 28), (401408, 784, 28, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf498 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_40.run(buf399, buf498, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf494, (401408, 512), (512, 1), 0), reinterpret_tensor(buf498, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf500 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(primals_92, buf500, 65536, stream=stream0)
        del primals_92
        buf501 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf502 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf503 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf493, (401408, 512), (512, 1), 0), buf501, buf502, buf503, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf507 = reinterpret_tensor(buf495, (512, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf495  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_58.run(buf493, buf507, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf507, buf500, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf509 = buf420; del buf420  # reuse
        buf510 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf542 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf543 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf376, buf509, buf510, buf542, buf543, 128, stream=stream0)
        buf511 = reinterpret_tensor(buf444, (512, 128, 784), (100352, 784, 1), 0); del buf444  # reuse
        buf517 = reinterpret_tensor(buf430, (512, 128, 784), (100352, 784, 1), 0); del buf430  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf508, buf511, buf517, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf511, buf509, buf510, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf514 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf518 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44.run(buf510, buf509, buf514, buf518, 128, stream=stream0)
        buf515 = buf511; del buf511  # reuse
        buf516 = reinterpret_tensor(buf508, (512, 128, 784), (100352, 784, 1), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf517, buf518, buf514, primals_94, primals_95, buf515, buf516, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del buf517
        del primals_95
        buf521 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf522 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf523 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf516, (100352, 512), (512, 1), 0), buf521, buf522, buf523, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf527 = reinterpret_tensor(buf516, (512, 128, 28, 28), (100352, 784, 28, 1), 0); del buf516  # reuse
        buf528 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf561 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_59.run(buf429, buf528, buf561, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf515, (512, 128, 28, 28), (100352, 784, 28, 1), 0), buf527, reinterpret_tensor(buf528, (512, 128, 28, 28), (100352, 784, 28, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf531 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        buf564 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_60.run(buf434, buf531, buf564, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf528, (100352, 512), (512, 1), 0), reinterpret_tensor(buf531, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf533 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_1_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_41.run(primals_98, buf533, 16384, 9, stream=stream0)
        del primals_98
        buf534 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf535 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf536 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf527, (100352, 512), (512, 1), 0), buf534, buf535, buf536, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf540 = reinterpret_tensor(buf515, (512, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf515  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf527, buf540, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_1_conv2], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, buf533, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf544 = reinterpret_tensor(buf540, (512, 128, 784), (100352, 784, 1), 0); del buf540  # reuse
        buf550 = reinterpret_tensor(buf527, (512, 128, 784), (100352, 784, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf541, buf544, buf550, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf544, buf542, buf543, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf547 = buf518; del buf518  # reuse
        buf551 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44.run(buf543, buf542, buf547, buf551, 128, stream=stream0)
        buf548 = buf544; del buf544  # reuse
        buf549 = reinterpret_tensor(buf541, (512, 128, 784), (100352, 784, 1), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf550, buf551, buf547, primals_100, primals_101, buf548, buf549, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del buf550
        del primals_101
        buf554 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf555 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf556 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf549, (100352, 512), (512, 1), 0), buf554, buf555, buf556, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf560 = reinterpret_tensor(buf549, (512, 128, 28, 28), (100352, 784, 28, 1), 0); del buf549  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf548, (512, 128, 28, 28), (100352, 784, 28, 1), 0), buf560, reinterpret_tensor(buf561, (512, 128, 28, 28), (100352, 784, 28, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf561, (100352, 512), (512, 1), 0), reinterpret_tensor(buf564, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf566 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_1_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(primals_104, buf566, 65536, stream=stream0)
        del primals_104
        buf567 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf568 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf569 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf560, (100352, 512), (512, 1), 0), buf567, buf568, buf569, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf573 = reinterpret_tensor(buf548, (512, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf548  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf560, buf573, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_1_conv3], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, buf566, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf575 = buf483; del buf483  # reuse
        buf576 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_61.run(buf446, buf575, buf576, 512, stream=stream0)
        buf577 = reinterpret_tensor(buf507, (512, 512, 784), (401408, 784, 1), 0); del buf507  # reuse
        buf583 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_53.run(buf574, buf577, buf583, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf577, buf575, buf576, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf580 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf584 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_54.run(buf576, buf575, buf580, buf584, 512, stream=stream0)
        buf581 = buf577; del buf577  # reuse
        buf582 = reinterpret_tensor(buf574, (512, 512, 784), (401408, 784, 1), 0); del buf574  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf583, buf584, buf580, primals_106, primals_107, buf581, buf582, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_107
        buf587 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf588 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf589 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf582, (401408, 512), (512, 1), 0), buf587, buf588, buf589, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf593 = reinterpret_tensor(buf582, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf582  # reuse
        buf594 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_56.run(buf492, buf594, 205520896, stream=stream0)
        buf595 = reinterpret_tensor(buf583, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf581, buf493, buf595, 205520896, stream=stream0)
        del buf493
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf595, buf593, reinterpret_tensor(buf594, (512, 512, 28, 28), (401408, 784, 28, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf598 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_40.run(buf399, buf598, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf594, (401408, 512), (512, 1), 0), reinterpret_tensor(buf598, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf600 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(primals_110, buf600, 65536, stream=stream0)
        del primals_110
        buf601 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf602 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf603 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf593, (401408, 512), (512, 1), 0), buf601, buf602, buf603, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf607 = reinterpret_tensor(buf595, (512, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_58.run(buf593, buf607, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten.convolution]
        buf608 = extern_kernels.convolution(buf607, buf600, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf608, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf609 = buf551; del buf551  # reuse
        buf610 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf642 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf643 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf376, buf609, buf610, buf642, buf643, 128, stream=stream0)
        buf611 = reinterpret_tensor(buf573, (512, 128, 784), (100352, 784, 1), 0); del buf573  # reuse
        buf617 = reinterpret_tensor(buf560, (512, 128, 784), (100352, 784, 1), 0); del buf560  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf608, buf611, buf617, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf611, buf609, buf610, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf614 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf618 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44.run(buf610, buf609, buf614, buf618, 128, stream=stream0)
        buf615 = buf611; del buf611  # reuse
        buf616 = reinterpret_tensor(buf608, (512, 128, 784), (100352, 784, 1), 0); del buf608  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf617, buf618, buf614, primals_112, primals_113, buf615, buf616, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del buf617
        del primals_113
        buf621 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf622 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf623 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf616, (100352, 512), (512, 1), 0), buf621, buf622, buf623, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf627 = reinterpret_tensor(buf616, (512, 128, 28, 28), (100352, 784, 28, 1), 0); del buf616  # reuse
        buf628 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf661 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_59.run(buf429, buf628, buf661, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf615, (512, 128, 28, 28), (100352, 784, 28, 1), 0), buf627, reinterpret_tensor(buf628, (512, 128, 28, 28), (100352, 784, 28, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf631 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        buf664 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_60.run(buf434, buf631, buf664, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf628, (100352, 512), (512, 1), 0), reinterpret_tensor(buf631, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf633 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_2_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_41.run(primals_116, buf633, 16384, 9, stream=stream0)
        del primals_116
        buf634 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf635 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf636 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf627, (100352, 512), (512, 1), 0), buf634, buf635, buf636, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf640 = reinterpret_tensor(buf615, (512, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf615  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf627, buf640, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_2_conv2], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(buf640, buf633, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf641, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf644 = reinterpret_tensor(buf640, (512, 128, 784), (100352, 784, 1), 0); del buf640  # reuse
        buf650 = reinterpret_tensor(buf627, (512, 128, 784), (100352, 784, 1), 0); del buf627  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf641, buf644, buf650, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf644, buf642, buf643, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf647 = buf618; del buf618  # reuse
        buf651 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44.run(buf643, buf642, buf647, buf651, 128, stream=stream0)
        buf648 = buf644; del buf644  # reuse
        buf649 = reinterpret_tensor(buf641, (512, 128, 784), (100352, 784, 1), 0); del buf641  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf650, buf651, buf647, primals_118, primals_119, buf648, buf649, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del buf650
        del primals_119
        buf654 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf655 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf656 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf649, (100352, 512), (512, 1), 0), buf654, buf655, buf656, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf660 = reinterpret_tensor(buf649, (512, 128, 28, 28), (100352, 784, 28, 1), 0); del buf649  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf648, (512, 128, 28, 28), (100352, 784, 28, 1), 0), buf660, reinterpret_tensor(buf661, (512, 128, 28, 28), (100352, 784, 28, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf661, (100352, 512), (512, 1), 0), reinterpret_tensor(buf664, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf666 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_2_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(primals_122, buf666, 65536, stream=stream0)
        del primals_122
        buf667 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf668 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf669 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf660, (100352, 512), (512, 1), 0), buf667, buf668, buf669, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf673 = reinterpret_tensor(buf648, (512, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf648  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf660, buf673, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_2_conv3], Original ATen: [aten.convolution]
        buf674 = extern_kernels.convolution(buf673, buf666, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf674, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf675 = buf584; del buf584  # reuse
        buf676 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_61.run(buf446, buf675, buf676, 512, stream=stream0)
        buf677 = reinterpret_tensor(buf607, (512, 512, 784), (401408, 784, 1), 0); del buf607  # reuse
        buf683 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_53.run(buf674, buf677, buf683, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf677, buf675, buf676, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf680 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf684 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_54.run(buf676, buf675, buf680, buf684, 512, stream=stream0)
        buf681 = buf677; del buf677  # reuse
        buf682 = reinterpret_tensor(buf674, (512, 512, 784), (401408, 784, 1), 0); del buf674  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf683, buf684, buf680, primals_124, primals_125, buf681, buf682, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_125
        buf687 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf688 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf689 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf682, (401408, 512), (512, 1), 0), buf687, buf688, buf689, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf693 = reinterpret_tensor(buf682, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf682  # reuse
        buf694 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_56.run(buf492, buf694, 205520896, stream=stream0)
        buf695 = reinterpret_tensor(buf683, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf683  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf681, buf593, buf695, 205520896, stream=stream0)
        del buf593
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf695, buf693, reinterpret_tensor(buf694, (512, 512, 28, 28), (401408, 784, 28, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf698 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_40.run(buf399, buf698, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf694, (401408, 512), (512, 1), 0), reinterpret_tensor(buf698, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf700 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(primals_128, buf700, 65536, stream=stream0)
        del primals_128
        buf701 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf702 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf703 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf693, (401408, 512), (512, 1), 0), buf701, buf702, buf703, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf707 = reinterpret_tensor(buf695, (512, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf695  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_58.run(buf693, buf707, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten.convolution]
        buf708 = extern_kernels.convolution(buf707, buf700, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf708, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf709 = buf651; del buf651  # reuse
        buf710 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf742 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_62.run(buf376, buf709, buf710, buf742, 128, stream=stream0)
        buf711 = reinterpret_tensor(buf673, (512, 128, 784), (100352, 784, 1), 0); del buf673  # reuse
        buf717 = reinterpret_tensor(buf660, (512, 128, 784), (100352, 784, 1), 0); del buf660  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf708, buf711, buf717, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf711, buf709, buf710, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf714 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf718 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44.run(buf710, buf709, buf714, buf718, 128, stream=stream0)
        buf715 = buf711; del buf711  # reuse
        buf716 = reinterpret_tensor(buf708, (512, 128, 784), (100352, 784, 1), 0); del buf708  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf717, buf718, buf714, primals_130, primals_131, buf715, buf716, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del buf717
        del primals_131
        buf721 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf722 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf723 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf716, (100352, 512), (512, 1), 0), buf721, buf722, buf723, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf727 = reinterpret_tensor(buf716, (512, 128, 28, 28), (100352, 784, 28, 1), 0); del buf716  # reuse
        buf728 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_46.run(buf429, buf728, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf715, (512, 128, 28, 28), (100352, 784, 28, 1), 0), buf727, reinterpret_tensor(buf728, (512, 128, 28, 28), (100352, 784, 28, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf731 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        buf762 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_60.run(buf434, buf731, buf762, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf728, (100352, 512), (512, 1), 0), reinterpret_tensor(buf731, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf733 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_3_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_41.run(primals_134, buf733, 16384, 9, stream=stream0)
        del primals_134
        buf734 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf735 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf736 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf727, (100352, 512), (512, 1), 0), buf734, buf735, buf736, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf740 = reinterpret_tensor(buf715, (512, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf715  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf727, buf740, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_3_conv2], Original ATen: [aten.convolution]
        buf741 = extern_kernels.convolution(buf740, buf733, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf741, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf743 = reinterpret_tensor(buf740, (512, 128, 784), (100352, 784, 1), 0); del buf740  # reuse
        buf749 = reinterpret_tensor(buf727, (512, 128, 784), (100352, 784, 1), 0); del buf727  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf741, buf743, buf749, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf743, buf376, buf742, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf746 = buf718; del buf718  # reuse
        buf750 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_44.run(buf742, buf376, buf746, buf750, 128, stream=stream0)
        buf747 = buf743; del buf743  # reuse
        buf748 = reinterpret_tensor(buf741, (512, 128, 784), (100352, 784, 1), 0); del buf741  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf749, buf750, buf746, primals_136, primals_137, buf747, buf748, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del buf749
        del buf750
        del primals_137
        buf753 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf754 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf755 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf748, (100352, 512), (512, 1), 0), buf753, buf754, buf755, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf759 = reinterpret_tensor(buf748, (512, 128, 28, 28), (100352, 784, 28, 1), 0); del buf748  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf747, (512, 128, 28, 28), (100352, 784, 28, 1), 0), buf759, buf429, 51380224, 1024, 50176, 1, 1, stream=stream0)
        del buf431
        del buf528
        del buf561
        del buf628
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf429, (100352, 512), (512, 1), 0), reinterpret_tensor(buf762, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf764 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_3_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(primals_140, buf764, 65536, stream=stream0)
        del primals_140
        buf765 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf766 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf767 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf759, (100352, 512), (512, 1), 0), buf765, buf766, buf767, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf771 = reinterpret_tensor(buf747, (512, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf747  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_50.run(buf759, buf771, 65536, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_3_conv3], Original ATen: [aten.convolution]
        buf772 = extern_kernels.convolution(buf771, buf764, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf772, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf773 = buf684; del buf684  # reuse
        buf774 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_61.run(buf446, buf773, buf774, 512, stream=stream0)
        buf775 = reinterpret_tensor(buf707, (512, 512, 784), (401408, 784, 1), 0); del buf707  # reuse
        buf781 = buf681; del buf681  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_53.run(buf772, buf775, buf781, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf775, buf773, buf774, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf778 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf782 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_54.run(buf774, buf773, buf778, buf782, 512, stream=stream0)
        buf779 = buf775; del buf775  # reuse
        buf780 = reinterpret_tensor(buf772, (512, 512, 784), (401408, 784, 1), 0); del buf772  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf781, buf782, buf778, primals_142, primals_143, buf779, buf780, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_143
        buf785 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf786 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf787 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf780, (401408, 512), (512, 1), 0), buf785, buf786, buf787, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf791 = reinterpret_tensor(buf780, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf780  # reuse
        buf792 = reinterpret_tensor(buf781, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf781  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf779, buf693, buf792, 205520896, stream=stream0)
        del buf693
        del buf779
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf792, buf791, buf492, 205520896, 1024, 200704, 1, 1, stream=stream0)
        del buf494
        del buf594
        del buf694
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf492, (401408, 512), (512, 1), 0), buf399, 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del buf492
        buf796 = empty_strided_cuda((256, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_55.run(primals_146, buf796, 131072, stream=stream0)
        del primals_146
        buf797 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf798 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf799 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf791, (401408, 512), (512, 1), 0), buf797, buf798, buf799, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf803 = reinterpret_tensor(buf792, (512, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf792  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_58.run(buf791, buf803, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten.convolution]
        buf804 = extern_kernels.convolution(buf803, buf796, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf804, (512, 256, 28, 28), (200704, 1, 7168, 256), 'torch.ops.aten.convolution.default')
        buf805 = buf353; del buf353  # reuse
        buf806 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf838 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf839 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_23.run(buf117, buf805, buf806, buf838, buf839, 256, stream=stream0)
        buf807 = reinterpret_tensor(buf342, (512, 256, 784), (200704, 784, 1), 0); del buf342  # reuse
        buf813 = reinterpret_tensor(buf330, (512, 256, 784), (200704, 784, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_63.run(buf804, buf807, buf813, 131072, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_15.run(buf807, buf805, buf806, 401408, 784, 200704, 784, 1024, 256, 392, 1, stream=stream0)
        buf810 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf814 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer3_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_64.run(buf806, buf805, buf810, buf814, 256, stream=stream0)
        buf811 = buf807; del buf807  # reuse
        buf812 = reinterpret_tensor(buf804, (512, 256, 784), (200704, 784, 1), 0); del buf804  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_16.run(buf813, buf814, buf810, primals_148, primals_149, buf811, buf812, 401408, 784, 200704, 784, 1024, 256, 392, 1, stream=stream0)
        del buf813
        del primals_149
        buf817 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf818 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf819 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf812, (200704, 512), (512, 1), 0), buf817, buf818, buf819, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf823 = reinterpret_tensor(buf812, (512, 256, 28, 28), (200704, 784, 28, 1), 0); del buf812  # reuse
        buf824 = reinterpret_tensor(buf67, (512, 256, 28, 28), (200704, 784, 28, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_15.run(buf824, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf811, (512, 256, 28, 28), (200704, 784, 28, 1), 0), buf823, buf824, 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf827 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_65.run(buf72, buf827, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf824, (200704, 512), (512, 1), 0), reinterpret_tensor(buf827, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf829 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_66.run(primals_152, buf829, 65536, 9, stream=stream0)
        del primals_152
        buf830 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf831 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf832 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf823, (200704, 512), (512, 1), 0), buf830, buf831, buf832, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf836 = reinterpret_tensor(buf811, (512, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf811  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_67.run(buf823, buf836, 131072, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten.convolution]
        buf837 = extern_kernels.convolution(buf836, buf829, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf837, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf840 = empty_strided_cuda((512, 256, 196), (50176, 196, 1), torch.bfloat16)
        buf846 = empty_strided_cuda((512, 256, 196), (50176, 196, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf837, buf840, buf846, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf840, buf838, buf839, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf843 = buf814; del buf814  # reuse
        buf847 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf839, buf838, buf843, buf847, 256, stream=stream0)
        buf844 = buf840; del buf840  # reuse
        buf845 = reinterpret_tensor(buf837, (512, 256, 196), (50176, 196, 1), 0); del buf837  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf846, buf847, buf843, primals_154, primals_155, buf844, buf845, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf846
        del primals_155
        buf850 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf851 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf852 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf845, (50176, 512), (512, 1), 0), buf850, buf851, buf852, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf856 = empty_strided_cuda((512, 256, 14, 14), (50176, 196, 14, 1), torch.int8)
        # Topologically Sorted Source Nodes: [layer3_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_70.run(buf856, 25690112, stream=stream0)
        buf857 = reinterpret_tensor(buf845, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf845  # reuse
        buf858 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_71.run(buf856, buf858, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf844, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf857, reinterpret_tensor(buf858, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        buf861 = empty_strided_cuda((50176, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer3_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_72.run(buf861, 802816, stream=stream0)
        buf862 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_73.run(buf861, buf862, 802816, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf858, (50176, 512), (512, 1), 0), reinterpret_tensor(buf862, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf864 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_158, buf864, 262144, stream=stream0)
        del primals_158
        buf865 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf866 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf867 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf857, (50176, 512), (512, 1), 0), buf865, buf866, buf867, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf871 = reinterpret_tensor(buf844, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf844  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf857, buf871, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten.convolution]
        buf872 = extern_kernels.convolution(buf871, buf864, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf872, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf873 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_76.run(buf873, 1024, stream=stream0)
        buf874 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf875 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf901 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf902 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_77.run(buf873, buf874, buf875, buf901, buf902, 1024, stream=stream0)
        buf876 = reinterpret_tensor(buf836, (512, 1024, 196), (200704, 196, 1), 0); del buf836  # reuse
        buf882 = reinterpret_tensor(buf823, (512, 1024, 196), (200704, 196, 1), 0); del buf823  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf872, buf876, buf882, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf876, buf874, buf875, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf879 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf883 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79.run(buf875, buf874, buf879, buf883, 1024, stream=stream0)
        buf880 = buf876; del buf876  # reuse
        buf881 = reinterpret_tensor(buf872, (512, 1024, 196), (200704, 196, 1), 0); del buf872  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf882, buf883, buf879, primals_160, primals_161, buf880, buf881, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_161
        buf886 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf887 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf888 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf881, (200704, 512), (512, 1), 0), buf886, buf887, buf888, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf892 = empty_strided_cuda((1024, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(primals_164, buf892, 524288, stream=stream0)
        del primals_164
        buf893 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf894 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf895 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf791, (401408, 512), (512, 1), 0), buf893, buf894, buf895, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf899 = buf803; del buf803  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_58.run(buf791, buf899, 262144, 784, stream=stream0)
        del buf791
        # Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten.convolution]
        buf900 = extern_kernels.convolution(buf899, buf892, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf900, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        del buf899
        buf903 = buf881; del buf881  # reuse
        buf909 = buf882; del buf882  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf900, buf903, buf909, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf903, buf901, buf902, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf906 = buf883; del buf883  # reuse
        buf910 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_downsample_1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79.run(buf902, buf901, buf906, buf910, 1024, stream=stream0)
        buf907 = buf903; del buf903  # reuse
        buf908 = reinterpret_tensor(buf900, (512, 1024, 196), (200704, 196, 1), 0); del buf900  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf909, buf910, buf906, primals_166, primals_167, buf907, buf908, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_167
        buf913 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf914 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf915 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf908, (200704, 512), (512, 1), 0), buf913, buf914, buf915, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf919 = reinterpret_tensor(buf824, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf824  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_relu_2], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_15.run(buf919, 102760448, stream=stream0)
        buf920 = reinterpret_tensor(buf908, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf908  # reuse
        buf921 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf919, buf921, 102760448, stream=stream0)
        buf922 = reinterpret_tensor(buf909, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf909  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_81.run(buf880, buf907, buf922, 102760448, stream=stream0)
        del buf880
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf922, buf920, reinterpret_tensor(buf921, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf925 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_65.run(buf72, buf925, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf921, (200704, 512), (512, 1), 0), reinterpret_tensor(buf925, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf927 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_170, buf927, 262144, stream=stream0)
        del primals_170
        buf928 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf929 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf930 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf920, (200704, 512), (512, 1), 0), buf928, buf929, buf930, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf934 = reinterpret_tensor(buf922, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf922  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_82.run(buf920, buf934, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten.convolution]
        buf935 = extern_kernels.convolution(buf934, buf927, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf935, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf936 = buf847; del buf847  # reuse
        buf937 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf969 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf970 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_23.run(buf117, buf936, buf937, buf969, buf970, 256, stream=stream0)
        buf938 = reinterpret_tensor(buf871, (512, 256, 196), (50176, 196, 1), 0); del buf871  # reuse
        buf944 = reinterpret_tensor(buf857, (512, 256, 196), (50176, 196, 1), 0); del buf857  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf935, buf938, buf944, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf938, buf936, buf937, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf941 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf945 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf937, buf936, buf941, buf945, 256, stream=stream0)
        buf942 = buf938; del buf938  # reuse
        buf943 = reinterpret_tensor(buf935, (512, 256, 196), (50176, 196, 1), 0); del buf935  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf944, buf945, buf941, primals_172, primals_173, buf942, buf943, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf944
        del primals_173
        buf948 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf949 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf950 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf943, (50176, 512), (512, 1), 0), buf948, buf949, buf950, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf954 = reinterpret_tensor(buf943, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf943  # reuse
        buf955 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf988 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_83.run(buf856, buf955, buf988, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf942, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf954, reinterpret_tensor(buf955, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        buf958 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        buf991 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_84.run(buf861, buf958, buf991, 802816, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf955, (50176, 512), (512, 1), 0), reinterpret_tensor(buf958, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf960 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_1_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_66.run(primals_176, buf960, 65536, 9, stream=stream0)
        del primals_176
        buf961 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf962 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf963 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf954, (50176, 512), (512, 1), 0), buf961, buf962, buf963, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf967 = reinterpret_tensor(buf942, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf942  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf954, buf967, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_1_conv2], Original ATen: [aten.convolution]
        buf968 = extern_kernels.convolution(buf967, buf960, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf968, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf971 = reinterpret_tensor(buf967, (512, 256, 196), (50176, 196, 1), 0); del buf967  # reuse
        buf977 = reinterpret_tensor(buf954, (512, 256, 196), (50176, 196, 1), 0); del buf954  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf968, buf971, buf977, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf971, buf969, buf970, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf974 = buf945; del buf945  # reuse
        buf978 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf970, buf969, buf974, buf978, 256, stream=stream0)
        buf975 = buf971; del buf971  # reuse
        buf976 = reinterpret_tensor(buf968, (512, 256, 196), (50176, 196, 1), 0); del buf968  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf977, buf978, buf974, primals_178, primals_179, buf975, buf976, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf977
        del primals_179
        buf981 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf982 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf983 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf976, (50176, 512), (512, 1), 0), buf981, buf982, buf983, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf987 = reinterpret_tensor(buf976, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf976  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf975, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf987, reinterpret_tensor(buf988, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf988, (50176, 512), (512, 1), 0), reinterpret_tensor(buf991, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf993 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_1_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_182, buf993, 262144, stream=stream0)
        del primals_182
        buf994 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf995 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf996 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf987, (50176, 512), (512, 1), 0), buf994, buf995, buf996, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1000 = reinterpret_tensor(buf975, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf975  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf987, buf1000, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_1_conv3], Original ATen: [aten.convolution]
        buf1001 = extern_kernels.convolution(buf1000, buf993, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1001, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf1002 = buf910; del buf910  # reuse
        buf1003 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_85.run(buf873, buf1002, buf1003, 1024, stream=stream0)
        buf1004 = reinterpret_tensor(buf934, (512, 1024, 196), (200704, 196, 1), 0); del buf934  # reuse
        buf1010 = buf907; del buf907  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf1001, buf1004, buf1010, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1004, buf1002, buf1003, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf1007 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1011 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79.run(buf1003, buf1002, buf1007, buf1011, 1024, stream=stream0)
        buf1008 = buf1004; del buf1004  # reuse
        buf1009 = reinterpret_tensor(buf1001, (512, 1024, 196), (200704, 196, 1), 0); del buf1001  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1010, buf1011, buf1007, primals_184, primals_185, buf1008, buf1009, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_185
        buf1014 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1015 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1016 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1009, (200704, 512), (512, 1), 0), buf1014, buf1015, buf1016, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1020 = reinterpret_tensor(buf1009, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1009  # reuse
        buf1021 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf919, buf1021, 102760448, stream=stream0)
        buf1022 = reinterpret_tensor(buf1010, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1010  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_81.run(buf1008, buf920, buf1022, 102760448, stream=stream0)
        del buf1008
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1022, buf1020, reinterpret_tensor(buf1021, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf1025 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_65.run(buf72, buf1025, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1021, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1025, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1027 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_188, buf1027, 262144, stream=stream0)
        del primals_188
        buf1028 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1029 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1030 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1020, (200704, 512), (512, 1), 0), buf1028, buf1029, buf1030, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1034 = reinterpret_tensor(buf1022, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1022  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_82.run(buf1020, buf1034, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten.convolution]
        buf1035 = extern_kernels.convolution(buf1034, buf1027, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1035, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1036 = buf978; del buf978  # reuse
        buf1037 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1069 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1070 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_23.run(buf117, buf1036, buf1037, buf1069, buf1070, 256, stream=stream0)
        buf1038 = reinterpret_tensor(buf1000, (512, 256, 196), (50176, 196, 1), 0); del buf1000  # reuse
        buf1044 = reinterpret_tensor(buf987, (512, 256, 196), (50176, 196, 1), 0); del buf987  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf1035, buf1038, buf1044, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1038, buf1036, buf1037, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf1041 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1045 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf1037, buf1036, buf1041, buf1045, 256, stream=stream0)
        buf1042 = buf1038; del buf1038  # reuse
        buf1043 = reinterpret_tensor(buf1035, (512, 256, 196), (50176, 196, 1), 0); del buf1035  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1044, buf1045, buf1041, primals_190, primals_191, buf1042, buf1043, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf1044
        del primals_191
        buf1048 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1049 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1050 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1043, (50176, 512), (512, 1), 0), buf1048, buf1049, buf1050, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1054 = reinterpret_tensor(buf1043, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1043  # reuse
        buf1055 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf1088 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_83.run(buf856, buf1055, buf1088, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1042, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf1054, reinterpret_tensor(buf1055, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        buf1058 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        buf1091 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_84.run(buf861, buf1058, buf1091, 802816, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1055, (50176, 512), (512, 1), 0), reinterpret_tensor(buf1058, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf1060 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_2_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_66.run(primals_194, buf1060, 65536, 9, stream=stream0)
        del primals_194
        buf1061 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1062 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1063 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1054, (50176, 512), (512, 1), 0), buf1061, buf1062, buf1063, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1067 = reinterpret_tensor(buf1042, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1042  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf1054, buf1067, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_2_conv2], Original ATen: [aten.convolution]
        buf1068 = extern_kernels.convolution(buf1067, buf1060, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1068, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1071 = reinterpret_tensor(buf1067, (512, 256, 196), (50176, 196, 1), 0); del buf1067  # reuse
        buf1077 = reinterpret_tensor(buf1054, (512, 256, 196), (50176, 196, 1), 0); del buf1054  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf1068, buf1071, buf1077, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1071, buf1069, buf1070, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf1074 = buf1045; del buf1045  # reuse
        buf1078 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf1070, buf1069, buf1074, buf1078, 256, stream=stream0)
        buf1075 = buf1071; del buf1071  # reuse
        buf1076 = reinterpret_tensor(buf1068, (512, 256, 196), (50176, 196, 1), 0); del buf1068  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1077, buf1078, buf1074, primals_196, primals_197, buf1075, buf1076, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf1077
        del primals_197
        buf1081 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1082 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1083 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1076, (50176, 512), (512, 1), 0), buf1081, buf1082, buf1083, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1087 = reinterpret_tensor(buf1076, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1076  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1075, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf1087, reinterpret_tensor(buf1088, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1088, (50176, 512), (512, 1), 0), reinterpret_tensor(buf1091, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf1093 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_2_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_200, buf1093, 262144, stream=stream0)
        del primals_200
        buf1094 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1095 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1096 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1087, (50176, 512), (512, 1), 0), buf1094, buf1095, buf1096, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1100 = reinterpret_tensor(buf1075, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1075  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf1087, buf1100, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_2_conv3], Original ATen: [aten.convolution]
        buf1101 = extern_kernels.convolution(buf1100, buf1093, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1101, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf1102 = buf1011; del buf1011  # reuse
        buf1103 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_85.run(buf873, buf1102, buf1103, 1024, stream=stream0)
        buf1104 = reinterpret_tensor(buf1034, (512, 1024, 196), (200704, 196, 1), 0); del buf1034  # reuse
        buf1110 = reinterpret_tensor(buf920, (512, 1024, 196), (200704, 196, 1), 0); del buf920  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf1101, buf1104, buf1110, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1104, buf1102, buf1103, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf1107 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1111 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79.run(buf1103, buf1102, buf1107, buf1111, 1024, stream=stream0)
        buf1108 = buf1104; del buf1104  # reuse
        buf1109 = reinterpret_tensor(buf1101, (512, 1024, 196), (200704, 196, 1), 0); del buf1101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1110, buf1111, buf1107, primals_202, primals_203, buf1108, buf1109, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_203
        buf1114 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1115 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1116 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1109, (200704, 512), (512, 1), 0), buf1114, buf1115, buf1116, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1120 = reinterpret_tensor(buf1109, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1109  # reuse
        buf1121 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf919, buf1121, 102760448, stream=stream0)
        buf1122 = reinterpret_tensor(buf1110, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_81.run(buf1108, buf1020, buf1122, 102760448, stream=stream0)
        del buf1020
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1122, buf1120, reinterpret_tensor(buf1121, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf1125 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_65.run(buf72, buf1125, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1121, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1125, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1127 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_206, buf1127, 262144, stream=stream0)
        del primals_206
        buf1128 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1129 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1130 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1120, (200704, 512), (512, 1), 0), buf1128, buf1129, buf1130, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1134 = reinterpret_tensor(buf1122, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1122  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_82.run(buf1120, buf1134, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten.convolution]
        buf1135 = extern_kernels.convolution(buf1134, buf1127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1135, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1136 = buf1078; del buf1078  # reuse
        buf1137 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1169 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1170 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_23.run(buf117, buf1136, buf1137, buf1169, buf1170, 256, stream=stream0)
        buf1138 = reinterpret_tensor(buf1100, (512, 256, 196), (50176, 196, 1), 0); del buf1100  # reuse
        buf1144 = reinterpret_tensor(buf1087, (512, 256, 196), (50176, 196, 1), 0); del buf1087  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf1135, buf1138, buf1144, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1138, buf1136, buf1137, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf1141 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1145 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf1137, buf1136, buf1141, buf1145, 256, stream=stream0)
        buf1142 = buf1138; del buf1138  # reuse
        buf1143 = reinterpret_tensor(buf1135, (512, 256, 196), (50176, 196, 1), 0); del buf1135  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1144, buf1145, buf1141, primals_208, primals_209, buf1142, buf1143, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf1144
        del primals_209
        buf1148 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1149 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1150 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1143, (50176, 512), (512, 1), 0), buf1148, buf1149, buf1150, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1154 = reinterpret_tensor(buf1143, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1143  # reuse
        buf1155 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf1188 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_83.run(buf856, buf1155, buf1188, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1142, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf1154, reinterpret_tensor(buf1155, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        buf1158 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        buf1191 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_84.run(buf861, buf1158, buf1191, 802816, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1155, (50176, 512), (512, 1), 0), reinterpret_tensor(buf1158, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf1160 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_3_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_66.run(primals_212, buf1160, 65536, 9, stream=stream0)
        del primals_212
        buf1161 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1162 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1163 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1154, (50176, 512), (512, 1), 0), buf1161, buf1162, buf1163, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1167 = reinterpret_tensor(buf1142, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1142  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf1154, buf1167, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_3_conv2], Original ATen: [aten.convolution]
        buf1168 = extern_kernels.convolution(buf1167, buf1160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1168, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1171 = reinterpret_tensor(buf1167, (512, 256, 196), (50176, 196, 1), 0); del buf1167  # reuse
        buf1177 = reinterpret_tensor(buf1154, (512, 256, 196), (50176, 196, 1), 0); del buf1154  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf1168, buf1171, buf1177, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1171, buf1169, buf1170, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf1174 = buf1145; del buf1145  # reuse
        buf1178 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf1170, buf1169, buf1174, buf1178, 256, stream=stream0)
        buf1175 = buf1171; del buf1171  # reuse
        buf1176 = reinterpret_tensor(buf1168, (512, 256, 196), (50176, 196, 1), 0); del buf1168  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1177, buf1178, buf1174, primals_214, primals_215, buf1175, buf1176, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf1177
        del primals_215
        buf1181 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1182 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1183 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1176, (50176, 512), (512, 1), 0), buf1181, buf1182, buf1183, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1187 = reinterpret_tensor(buf1176, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1175, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf1187, reinterpret_tensor(buf1188, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1188, (50176, 512), (512, 1), 0), reinterpret_tensor(buf1191, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf1193 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_3_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_218, buf1193, 262144, stream=stream0)
        del primals_218
        buf1194 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1195 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1196 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1187, (50176, 512), (512, 1), 0), buf1194, buf1195, buf1196, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1200 = reinterpret_tensor(buf1175, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1175  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf1187, buf1200, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_3_conv3], Original ATen: [aten.convolution]
        buf1201 = extern_kernels.convolution(buf1200, buf1193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1201, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf1202 = buf1111; del buf1111  # reuse
        buf1203 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_85.run(buf873, buf1202, buf1203, 1024, stream=stream0)
        buf1204 = reinterpret_tensor(buf1134, (512, 1024, 196), (200704, 196, 1), 0); del buf1134  # reuse
        buf1210 = buf1108; del buf1108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf1201, buf1204, buf1210, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1204, buf1202, buf1203, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf1207 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1211 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79.run(buf1203, buf1202, buf1207, buf1211, 1024, stream=stream0)
        buf1208 = buf1204; del buf1204  # reuse
        buf1209 = reinterpret_tensor(buf1201, (512, 1024, 196), (200704, 196, 1), 0); del buf1201  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1210, buf1211, buf1207, primals_220, primals_221, buf1208, buf1209, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_221
        buf1214 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1215 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1216 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1209, (200704, 512), (512, 1), 0), buf1214, buf1215, buf1216, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1220 = reinterpret_tensor(buf1209, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1209  # reuse
        buf1221 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf919, buf1221, 102760448, stream=stream0)
        buf1222 = reinterpret_tensor(buf1210, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1210  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_81.run(buf1208, buf1120, buf1222, 102760448, stream=stream0)
        del buf1120
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1222, buf1220, reinterpret_tensor(buf1221, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf1225 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_65.run(buf72, buf1225, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1221, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1225, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1227 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_224, buf1227, 262144, stream=stream0)
        del primals_224
        buf1228 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1229 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1230 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1220, (200704, 512), (512, 1), 0), buf1228, buf1229, buf1230, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1234 = reinterpret_tensor(buf1222, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1222  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_82.run(buf1220, buf1234, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten.convolution]
        buf1235 = extern_kernels.convolution(buf1234, buf1227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1235, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1236 = buf1178; del buf1178  # reuse
        buf1237 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1269 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1270 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_23.run(buf117, buf1236, buf1237, buf1269, buf1270, 256, stream=stream0)
        buf1238 = reinterpret_tensor(buf1200, (512, 256, 196), (50176, 196, 1), 0); del buf1200  # reuse
        buf1244 = reinterpret_tensor(buf1187, (512, 256, 196), (50176, 196, 1), 0); del buf1187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf1235, buf1238, buf1244, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1238, buf1236, buf1237, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf1241 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1245 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf1237, buf1236, buf1241, buf1245, 256, stream=stream0)
        buf1242 = buf1238; del buf1238  # reuse
        buf1243 = reinterpret_tensor(buf1235, (512, 256, 196), (50176, 196, 1), 0); del buf1235  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1244, buf1245, buf1241, primals_226, primals_227, buf1242, buf1243, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf1244
        del primals_227
        buf1248 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1249 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1250 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1243, (50176, 512), (512, 1), 0), buf1248, buf1249, buf1250, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1254 = reinterpret_tensor(buf1243, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1243  # reuse
        buf1255 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf1288 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_83.run(buf856, buf1255, buf1288, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1242, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf1254, reinterpret_tensor(buf1255, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        buf1258 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        buf1291 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_84.run(buf861, buf1258, buf1291, 802816, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1255, (50176, 512), (512, 1), 0), reinterpret_tensor(buf1258, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf1260 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_4_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_66.run(primals_230, buf1260, 65536, 9, stream=stream0)
        del primals_230
        buf1261 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1262 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1263 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1254, (50176, 512), (512, 1), 0), buf1261, buf1262, buf1263, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1267 = reinterpret_tensor(buf1242, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1242  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf1254, buf1267, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_4_conv2], Original ATen: [aten.convolution]
        buf1268 = extern_kernels.convolution(buf1267, buf1260, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1268, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1271 = reinterpret_tensor(buf1267, (512, 256, 196), (50176, 196, 1), 0); del buf1267  # reuse
        buf1277 = reinterpret_tensor(buf1254, (512, 256, 196), (50176, 196, 1), 0); del buf1254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf1268, buf1271, buf1277, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1271, buf1269, buf1270, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf1274 = buf1245; del buf1245  # reuse
        buf1278 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf1270, buf1269, buf1274, buf1278, 256, stream=stream0)
        buf1275 = buf1271; del buf1271  # reuse
        buf1276 = reinterpret_tensor(buf1268, (512, 256, 196), (50176, 196, 1), 0); del buf1268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1277, buf1278, buf1274, primals_232, primals_233, buf1275, buf1276, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf1277
        del primals_233
        buf1281 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1282 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1283 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1276, (50176, 512), (512, 1), 0), buf1281, buf1282, buf1283, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1287 = reinterpret_tensor(buf1276, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1276  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1275, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf1287, reinterpret_tensor(buf1288, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1288, (50176, 512), (512, 1), 0), reinterpret_tensor(buf1291, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf1293 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_4_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_236, buf1293, 262144, stream=stream0)
        del primals_236
        buf1294 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1295 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1296 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1287, (50176, 512), (512, 1), 0), buf1294, buf1295, buf1296, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1300 = reinterpret_tensor(buf1275, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1275  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf1287, buf1300, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_4_conv3], Original ATen: [aten.convolution]
        buf1301 = extern_kernels.convolution(buf1300, buf1293, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1301, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf1302 = buf1211; del buf1211  # reuse
        buf1303 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_85.run(buf873, buf1302, buf1303, 1024, stream=stream0)
        buf1304 = reinterpret_tensor(buf1234, (512, 1024, 196), (200704, 196, 1), 0); del buf1234  # reuse
        buf1310 = buf1208; del buf1208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf1301, buf1304, buf1310, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1304, buf1302, buf1303, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf1307 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1311 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79.run(buf1303, buf1302, buf1307, buf1311, 1024, stream=stream0)
        buf1308 = buf1304; del buf1304  # reuse
        buf1309 = reinterpret_tensor(buf1301, (512, 1024, 196), (200704, 196, 1), 0); del buf1301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1310, buf1311, buf1307, primals_238, primals_239, buf1308, buf1309, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_239
        buf1314 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1315 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1316 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1309, (200704, 512), (512, 1), 0), buf1314, buf1315, buf1316, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1320 = reinterpret_tensor(buf1309, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1309  # reuse
        buf1321 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf919, buf1321, 102760448, stream=stream0)
        buf1322 = reinterpret_tensor(buf1310, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_81.run(buf1308, buf1220, buf1322, 102760448, stream=stream0)
        del buf1220
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1322, buf1320, reinterpret_tensor(buf1321, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf1325 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_65.run(buf72, buf1325, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1321, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1325, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1327 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_242, buf1327, 262144, stream=stream0)
        del primals_242
        buf1328 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1329 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1330 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1320, (200704, 512), (512, 1), 0), buf1328, buf1329, buf1330, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1334 = reinterpret_tensor(buf1322, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1322  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_82.run(buf1320, buf1334, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten.convolution]
        buf1335 = extern_kernels.convolution(buf1334, buf1327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1335, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1336 = buf1278; del buf1278  # reuse
        buf1337 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1369 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_86.run(buf117, buf1336, buf1337, buf1369, 256, stream=stream0)
        buf1338 = reinterpret_tensor(buf1300, (512, 256, 196), (50176, 196, 1), 0); del buf1300  # reuse
        buf1344 = reinterpret_tensor(buf1287, (512, 256, 196), (50176, 196, 1), 0); del buf1287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf1335, buf1338, buf1344, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1338, buf1336, buf1337, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf1341 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1345 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf1337, buf1336, buf1341, buf1345, 256, stream=stream0)
        buf1342 = buf1338; del buf1338  # reuse
        buf1343 = reinterpret_tensor(buf1335, (512, 256, 196), (50176, 196, 1), 0); del buf1335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1344, buf1345, buf1341, primals_244, primals_245, buf1342, buf1343, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf1344
        del primals_245
        buf1348 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1349 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1350 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1343, (50176, 512), (512, 1), 0), buf1348, buf1349, buf1350, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1354 = reinterpret_tensor(buf1343, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1343  # reuse
        buf1355 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_71.run(buf856, buf1355, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1342, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf1354, reinterpret_tensor(buf1355, (512, 256, 14, 14), (50176, 196, 14, 1), 0), 25690112, 1024, 25088, 1, 1, stream=stream0)
        buf1358 = empty_strided_cuda((802816, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_73.run(buf861, buf1358, 802816, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1355, (50176, 512), (512, 1), 0), reinterpret_tensor(buf1358, (50176, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        buf1360 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_5_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_66.run(primals_248, buf1360, 65536, 9, stream=stream0)
        del primals_248
        buf1361 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1362 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1363 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1354, (50176, 512), (512, 1), 0), buf1361, buf1362, buf1363, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1367 = reinterpret_tensor(buf1342, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1342  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf1354, buf1367, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_5_conv2], Original ATen: [aten.convolution]
        buf1368 = extern_kernels.convolution(buf1367, buf1360, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1368, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1370 = reinterpret_tensor(buf1367, (512, 256, 196), (50176, 196, 1), 0); del buf1367  # reuse
        buf1376 = reinterpret_tensor(buf1354, (512, 256, 196), (50176, 196, 1), 0); del buf1354  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf1368, buf1370, buf1376, 131072, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1370, buf117, buf1369, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf1373 = buf1345; del buf1345  # reuse
        buf1377 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_69.run(buf1369, buf117, buf1373, buf1377, 256, stream=stream0)
        buf1374 = buf1370; del buf1370  # reuse
        buf1375 = reinterpret_tensor(buf1368, (512, 256, 196), (50176, 196, 1), 0); del buf1368  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1376, buf1377, buf1373, primals_250, primals_251, buf1374, buf1375, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf1376
        del buf1377
        del primals_251
        buf1380 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1381 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1382 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1375, (50176, 512), (512, 1), 0), buf1380, buf1381, buf1382, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1386 = reinterpret_tensor(buf1375, (512, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1375  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1374, (512, 256, 14, 14), (50176, 196, 14, 1), 0), buf1386, buf856, 25690112, 1024, 25088, 1, 1, stream=stream0)
        del buf1055
        del buf1088
        del buf1155
        del buf1188
        del buf1255
        del buf1288
        del buf1355
        del buf858
        del buf955
        del buf988
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf856, (50176, 512), (512, 1), 0), buf861, 512, 1, 16, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del buf856
        buf1390 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_5_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(primals_254, buf1390, 262144, stream=stream0)
        del primals_254
        buf1391 = empty_strided_cuda((50176, 32), (32, 1), torch.int32)
        buf1392 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        buf1393 = empty_strided_cuda((50176, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1386, (50176, 512), (512, 1), 0), buf1391, buf1392, buf1393, 512, 1, 32, 1, 2, 16, 32, 3, 50176, 1, 1, stream=stream0)
        buf1397 = reinterpret_tensor(buf1374, (512, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1374  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_75.run(buf1386, buf1397, 131072, 196, stream=stream0)
        del buf1386
        # Topologically Sorted Source Nodes: [layer3_5_conv3], Original ATen: [aten.convolution]
        buf1398 = extern_kernels.convolution(buf1397, buf1390, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1398, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        del buf1397
        buf1399 = buf1311; del buf1311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_87.run(buf873, buf1399, 1024, stream=stream0)
        buf1400 = reinterpret_tensor(buf1334, (512, 1024, 196), (200704, 196, 1), 0); del buf1334  # reuse
        buf1406 = buf1308; del buf1308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf1398, buf1400, buf1406, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1400, buf873, buf1399, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf1403 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1407 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_79.run(buf1399, buf873, buf1403, buf1407, 1024, stream=stream0)
        buf1404 = buf1400; del buf1400  # reuse
        buf1405 = reinterpret_tensor(buf1398, (512, 1024, 196), (200704, 196, 1), 0); del buf1398  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1406, buf1407, buf1403, primals_256, primals_257, buf1404, buf1405, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del buf1407
        del primals_257
        buf1410 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1411 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1412 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1405, (200704, 512), (512, 1), 0), buf1410, buf1411, buf1412, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1416 = reinterpret_tensor(buf1405, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1405  # reuse
        buf1417 = reinterpret_tensor(buf1406, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1406  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_81.run(buf1404, buf1320, buf1417, 102760448, stream=stream0)
        del buf1320
        del buf1404
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1417, buf1416, buf919, 102760448, 1024, 100352, 1, 1, stream=stream0)
        del buf1021
        del buf1121
        del buf1221
        del buf1321
        del buf921
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf919, (200704, 512), (512, 1), 0), buf72, 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del buf919
        buf1421 = empty_strided_cuda((512, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(primals_260, buf1421, 524288, stream=stream0)
        del primals_260
        buf1422 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1423 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1424 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1416, (200704, 512), (512, 1), 0), buf1422, buf1423, buf1424, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1428 = reinterpret_tensor(buf1417, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1417  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_82.run(buf1416, buf1428, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten.convolution]
        buf1429 = extern_kernels.convolution(buf1428, buf1421, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1429, (512, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution.default')
        buf1430 = buf782; del buf782  # reuse
        buf1431 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1463 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1464 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_52.run(buf446, buf1430, buf1431, buf1463, buf1464, 512, stream=stream0)
        buf1432 = reinterpret_tensor(buf771, (512, 512, 196), (100352, 196, 1), 0); del buf771  # reuse
        buf1438 = reinterpret_tensor(buf759, (512, 512, 196), (100352, 196, 1), 0); del buf759  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_88.run(buf1429, buf1432, buf1438, 262144, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_21.run(buf1432, buf1430, buf1431, 100352, 196, 100352, 196, 1024, 512, 98, 1, stream=stream0)
        buf1435 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1439 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer4_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_89.run(buf1431, buf1430, buf1435, buf1439, 512, stream=stream0)
        buf1436 = buf1432; del buf1432  # reuse
        buf1437 = reinterpret_tensor(buf1429, (512, 512, 196), (100352, 196, 1), 0); del buf1429  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_22.run(buf1438, buf1439, buf1435, primals_262, primals_263, buf1436, buf1437, 100352, 196, 100352, 196, 1024, 512, 98, 1, stream=stream0)
        del buf1438
        del primals_263
        buf1442 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1443 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1444 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1437, (100352, 512), (512, 1), 0), buf1442, buf1443, buf1444, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1448 = reinterpret_tensor(buf1437, (512, 512, 14, 14), (100352, 196, 14, 1), 0); del buf1437  # reuse
        buf1449 = reinterpret_tensor(buf429, (512, 512, 14, 14), (100352, 196, 14, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_45.run(buf1449, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1436, (512, 512, 14, 14), (100352, 196, 14, 1), 0), buf1448, buf1449, 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf1452 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf434, buf1452, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1449, (100352, 512), (512, 1), 0), reinterpret_tensor(buf1452, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf1454 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_90.run(primals_266, buf1454, 262144, 9, stream=stream0)
        del primals_266
        buf1455 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1456 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1457 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1448, (100352, 512), (512, 1), 0), buf1455, buf1456, buf1457, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1461 = reinterpret_tensor(buf1436, (512, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf1436  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_91.run(buf1448, buf1461, 262144, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten.convolution]
        buf1462 = extern_kernels.convolution(buf1461, buf1454, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1462, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1465 = empty_strided_cuda((512, 512, 49), (25088, 49, 1), torch.bfloat16)
        buf1471 = empty_strided_cuda((512, 512, 49), (25088, 49, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_92.run(buf1462, buf1465, buf1471, 262144, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1465, buf1463, buf1464, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf1468 = buf1439; del buf1439  # reuse
        buf1472 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_93.run(buf1464, buf1463, buf1468, buf1472, 512, stream=stream0)
        buf1469 = buf1465; del buf1465  # reuse
        buf1470 = reinterpret_tensor(buf1462, (512, 512, 49), (25088, 49, 1), 0); del buf1462  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1471, buf1472, buf1468, primals_268, primals_269, buf1469, buf1470, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del buf1471
        del primals_269
        buf1475 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1476 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1477 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1470, (25088, 512), (512, 1), 0), buf1475, buf1476, buf1477, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1481 = empty_strided_cuda((512, 512, 7, 7), (25088, 49, 7, 1), torch.int8)
        # Topologically Sorted Source Nodes: [layer4_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_94.run(buf1481, 12845056, stream=stream0)
        buf1482 = reinterpret_tensor(buf1470, (512, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1470  # reuse
        buf1483 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_95.run(buf1481, buf1483, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1469, (512, 512, 7, 7), (25088, 49, 7, 1), 0), buf1482, reinterpret_tensor(buf1483, (512, 512, 7, 7), (25088, 49, 7, 1), 0), 12845056, 1024, 12544, 1, 1, stream=stream0)
        buf1486 = empty_strided_cuda((25088, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer4_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_96.run(buf1486, 401408, stream=stream0)
        buf1487 = empty_strided_cuda((401408, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_97.run(buf1486, buf1487, 401408, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1483, (25088, 512), (512, 1), 0), reinterpret_tensor(buf1487, (25088, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        buf1489 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_98.run(primals_272, buf1489, 1048576, stream=stream0)
        del primals_272
        buf1490 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1491 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1492 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1482, (25088, 512), (512, 1), 0), buf1490, buf1491, buf1492, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1496 = reinterpret_tensor(buf1469, (512, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1469  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_99.run(buf1482, buf1496, 262144, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten.convolution]
        buf1497 = extern_kernels.convolution(buf1496, buf1489, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1497, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        buf1498 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_100.run(buf1498, 2048, stream=stream0)
        buf1499 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1500 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1526 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1527 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_101.run(buf1498, buf1499, buf1500, buf1526, buf1527, 2048, stream=stream0)
        buf1501 = reinterpret_tensor(buf1461, (512, 2048, 49), (100352, 49, 1), 0); del buf1461  # reuse
        buf1507 = reinterpret_tensor(buf1448, (512, 2048, 49), (100352, 49, 1), 0); del buf1448  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_102.run(buf1497, buf1501, buf1507, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_25.run(buf1501, buf1499, buf1500, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf1504 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1508 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103.run(buf1500, buf1499, buf1504, buf1508, 2048, stream=stream0)
        buf1505 = buf1501; del buf1501  # reuse
        buf1506 = reinterpret_tensor(buf1497, (512, 2048, 49), (100352, 49, 1), 0); del buf1497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_26.run(buf1507, buf1508, buf1504, primals_274, primals_275, buf1505, buf1506, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del primals_275
        buf1511 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1512 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1513 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1506, (100352, 512), (512, 1), 0), buf1511, buf1512, buf1513, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1517 = empty_strided_cuda((2048, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_104.run(primals_278, buf1517, 2097152, stream=stream0)
        del primals_278
        buf1518 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1519 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1520 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1416, (200704, 512), (512, 1), 0), buf1518, buf1519, buf1520, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1524 = buf1428; del buf1428  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_82.run(buf1416, buf1524, 524288, 196, stream=stream0)
        del buf1416
        # Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten.convolution]
        buf1525 = extern_kernels.convolution(buf1524, buf1517, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1525, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        del buf1524
        buf1528 = buf1506; del buf1506  # reuse
        buf1534 = buf1507; del buf1507  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_102.run(buf1525, buf1528, buf1534, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_25.run(buf1528, buf1526, buf1527, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf1531 = buf1508; del buf1508  # reuse
        buf1535 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_downsample_1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103.run(buf1527, buf1526, buf1531, buf1535, 2048, stream=stream0)
        buf1532 = buf1528; del buf1528  # reuse
        buf1533 = reinterpret_tensor(buf1525, (512, 2048, 49), (100352, 49, 1), 0); del buf1525  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_26.run(buf1534, buf1535, buf1531, primals_280, primals_281, buf1532, buf1533, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del primals_281
        buf1538 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1539 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1540 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1533, (100352, 512), (512, 1), 0), buf1538, buf1539, buf1540, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1544 = reinterpret_tensor(buf1449, (512, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1449  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_relu_2], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_45.run(buf1544, 51380224, stream=stream0)
        buf1545 = reinterpret_tensor(buf1533, (512, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1533  # reuse
        buf1546 = buf728; del buf728  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_46.run(buf1544, buf1546, 51380224, stream=stream0)
        buf1547 = reinterpret_tensor(buf1534, (512, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1534  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_105.run(buf1505, buf1532, buf1547, 51380224, stream=stream0)
        del buf1505
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1547, buf1545, reinterpret_tensor(buf1546, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf1550 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf434, buf1550, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1546, (100352, 512), (512, 1), 0), reinterpret_tensor(buf1550, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf1552 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_98.run(primals_284, buf1552, 1048576, stream=stream0)
        del primals_284
        buf1553 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1554 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1555 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1545, (100352, 512), (512, 1), 0), buf1553, buf1554, buf1555, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1559 = reinterpret_tensor(buf1547, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1547  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_106.run(buf1545, buf1559, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten.convolution]
        buf1560 = extern_kernels.convolution(buf1559, buf1552, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1560, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1561 = buf1472; del buf1472  # reuse
        buf1562 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1594 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1595 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_52.run(buf446, buf1561, buf1562, buf1594, buf1595, 512, stream=stream0)
        buf1563 = reinterpret_tensor(buf1496, (512, 512, 49), (25088, 49, 1), 0); del buf1496  # reuse
        buf1569 = reinterpret_tensor(buf1482, (512, 512, 49), (25088, 49, 1), 0); del buf1482  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_92.run(buf1560, buf1563, buf1569, 262144, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1563, buf1561, buf1562, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf1566 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1570 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_93.run(buf1562, buf1561, buf1566, buf1570, 512, stream=stream0)
        buf1567 = buf1563; del buf1563  # reuse
        buf1568 = reinterpret_tensor(buf1560, (512, 512, 49), (25088, 49, 1), 0); del buf1560  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1569, buf1570, buf1566, primals_286, primals_287, buf1567, buf1568, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del buf1569
        del primals_287
        buf1573 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1574 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1575 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1568, (25088, 512), (512, 1), 0), buf1573, buf1574, buf1575, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1579 = reinterpret_tensor(buf1568, (512, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1568  # reuse
        buf1580 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        buf1613 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_107.run(buf1481, buf1580, buf1613, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1567, (512, 512, 7, 7), (25088, 49, 7, 1), 0), buf1579, reinterpret_tensor(buf1580, (512, 512, 7, 7), (25088, 49, 7, 1), 0), 12845056, 1024, 12544, 1, 1, stream=stream0)
        buf1583 = empty_strided_cuda((401408, ), (1, ), torch.int32)
        buf1616 = empty_strided_cuda((401408, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_108.run(buf1486, buf1583, buf1616, 401408, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1580, (25088, 512), (512, 1), 0), reinterpret_tensor(buf1583, (25088, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        buf1585 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_1_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_90.run(primals_290, buf1585, 262144, 9, stream=stream0)
        del primals_290
        buf1586 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1587 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1588 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1579, (25088, 512), (512, 1), 0), buf1586, buf1587, buf1588, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1592 = reinterpret_tensor(buf1567, (512, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1567  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_99.run(buf1579, buf1592, 262144, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_1_conv2], Original ATen: [aten.convolution]
        buf1593 = extern_kernels.convolution(buf1592, buf1585, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1593, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1596 = reinterpret_tensor(buf1592, (512, 512, 49), (25088, 49, 1), 0); del buf1592  # reuse
        buf1602 = reinterpret_tensor(buf1579, (512, 512, 49), (25088, 49, 1), 0); del buf1579  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_92.run(buf1593, buf1596, buf1602, 262144, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1596, buf1594, buf1595, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf1599 = buf1570; del buf1570  # reuse
        buf1603 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_93.run(buf1595, buf1594, buf1599, buf1603, 512, stream=stream0)
        buf1600 = buf1596; del buf1596  # reuse
        buf1601 = reinterpret_tensor(buf1593, (512, 512, 49), (25088, 49, 1), 0); del buf1593  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1602, buf1603, buf1599, primals_292, primals_293, buf1600, buf1601, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del buf1602
        del primals_293
        buf1606 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1607 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1608 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1601, (25088, 512), (512, 1), 0), buf1606, buf1607, buf1608, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1612 = reinterpret_tensor(buf1601, (512, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1601  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1600, (512, 512, 7, 7), (25088, 49, 7, 1), 0), buf1612, reinterpret_tensor(buf1613, (512, 512, 7, 7), (25088, 49, 7, 1), 0), 12845056, 1024, 12544, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1613, (25088, 512), (512, 1), 0), reinterpret_tensor(buf1616, (25088, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        buf1618 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_1_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_98.run(primals_296, buf1618, 1048576, stream=stream0)
        del primals_296
        buf1619 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1620 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1621 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1612, (25088, 512), (512, 1), 0), buf1619, buf1620, buf1621, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1625 = reinterpret_tensor(buf1600, (512, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1600  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_99.run(buf1612, buf1625, 262144, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_1_conv3], Original ATen: [aten.convolution]
        buf1626 = extern_kernels.convolution(buf1625, buf1618, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1626, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        buf1627 = buf1535; del buf1535  # reuse
        buf1628 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_109.run(buf1498, buf1627, buf1628, 2048, stream=stream0)
        buf1629 = reinterpret_tensor(buf1559, (512, 2048, 49), (100352, 49, 1), 0); del buf1559  # reuse
        buf1635 = buf1532; del buf1532  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_102.run(buf1626, buf1629, buf1635, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_25.run(buf1629, buf1627, buf1628, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf1632 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1636 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103.run(buf1628, buf1627, buf1632, buf1636, 2048, stream=stream0)
        buf1633 = buf1629; del buf1629  # reuse
        buf1634 = reinterpret_tensor(buf1626, (512, 2048, 49), (100352, 49, 1), 0); del buf1626  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_26.run(buf1635, buf1636, buf1632, primals_298, primals_299, buf1633, buf1634, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del primals_299
        buf1639 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1640 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1641 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1634, (100352, 512), (512, 1), 0), buf1639, buf1640, buf1641, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1645 = reinterpret_tensor(buf1634, (512, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1634  # reuse
        buf1646 = buf661; del buf661  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_46.run(buf1544, buf1646, 51380224, stream=stream0)
        buf1647 = reinterpret_tensor(buf1635, (512, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1635  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_105.run(buf1633, buf1545, buf1647, 51380224, stream=stream0)
        del buf1545
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1647, buf1645, reinterpret_tensor(buf1646, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf1650 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf434, buf1650, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1646, (100352, 512), (512, 1), 0), reinterpret_tensor(buf1650, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf1652 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_98.run(primals_302, buf1652, 1048576, stream=stream0)
        del primals_302
        buf1653 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1654 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1655 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1645, (100352, 512), (512, 1), 0), buf1653, buf1654, buf1655, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1659 = reinterpret_tensor(buf1647, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1647  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_106.run(buf1645, buf1659, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten.convolution]
        buf1660 = extern_kernels.convolution(buf1659, buf1652, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1660, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1661 = buf1603; del buf1603  # reuse
        buf1662 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1694 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_110.run(buf446, buf1661, buf1662, buf1694, 512, stream=stream0)
        buf1663 = reinterpret_tensor(buf1625, (512, 512, 49), (25088, 49, 1), 0); del buf1625  # reuse
        buf1669 = reinterpret_tensor(buf1612, (512, 512, 49), (25088, 49, 1), 0); del buf1612  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_92.run(buf1660, buf1663, buf1669, 262144, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1663, buf1661, buf1662, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf1666 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1670 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_93.run(buf1662, buf1661, buf1666, buf1670, 512, stream=stream0)
        buf1667 = buf1663; del buf1663  # reuse
        buf1668 = reinterpret_tensor(buf1660, (512, 512, 49), (25088, 49, 1), 0); del buf1660  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1669, buf1670, buf1666, primals_304, primals_305, buf1667, buf1668, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del buf1669
        del primals_305
        buf1673 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1674 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1675 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1668, (25088, 512), (512, 1), 0), buf1673, buf1674, buf1675, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1679 = reinterpret_tensor(buf1668, (512, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1668  # reuse
        buf1680 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_95.run(buf1481, buf1680, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1667, (512, 512, 7, 7), (25088, 49, 7, 1), 0), buf1679, reinterpret_tensor(buf1680, (512, 512, 7, 7), (25088, 49, 7, 1), 0), 12845056, 1024, 12544, 1, 1, stream=stream0)
        buf1683 = empty_strided_cuda((401408, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_97.run(buf1486, buf1683, 401408, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1680, (25088, 512), (512, 1), 0), reinterpret_tensor(buf1683, (25088, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        buf1685 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_2_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_90.run(primals_308, buf1685, 262144, 9, stream=stream0)
        del primals_308
        buf1686 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1687 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1688 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1679, (25088, 512), (512, 1), 0), buf1686, buf1687, buf1688, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1692 = reinterpret_tensor(buf1667, (512, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1667  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_99.run(buf1679, buf1692, 262144, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_2_conv2], Original ATen: [aten.convolution]
        buf1693 = extern_kernels.convolution(buf1692, buf1685, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1693, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1695 = reinterpret_tensor(buf1692, (512, 512, 49), (25088, 49, 1), 0); del buf1692  # reuse
        buf1701 = reinterpret_tensor(buf1679, (512, 512, 49), (25088, 49, 1), 0); del buf1679  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_92.run(buf1693, buf1695, buf1701, 262144, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1695, buf446, buf1694, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf1698 = buf1670; del buf1670  # reuse
        buf1702 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_93.run(buf1694, buf446, buf1698, buf1702, 512, stream=stream0)
        buf1699 = buf1695; del buf1695  # reuse
        buf1700 = reinterpret_tensor(buf1693, (512, 512, 49), (25088, 49, 1), 0); del buf1693  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1701, buf1702, buf1698, primals_310, primals_311, buf1699, buf1700, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del buf1701
        del buf1702
        del primals_311
        buf1705 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1706 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1707 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1700, (25088, 512), (512, 1), 0), buf1705, buf1706, buf1707, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1711 = reinterpret_tensor(buf1700, (512, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1700  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1699, (512, 512, 7, 7), (25088, 49, 7, 1), 0), buf1711, buf1481, 12845056, 1024, 12544, 1, 1, stream=stream0)
        del buf1483
        del buf1580
        del buf1613
        del buf1680
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1481, (25088, 512), (512, 1), 0), buf1486, 512, 1, 16, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        del buf1481
        buf1715 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_2_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_98.run(primals_314, buf1715, 1048576, stream=stream0)
        del primals_314
        buf1716 = empty_strided_cuda((25088, 32), (32, 1), torch.int32)
        buf1717 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        buf1718 = empty_strided_cuda((25088, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1711, (25088, 512), (512, 1), 0), buf1716, buf1717, buf1718, 512, 1, 32, 1, 2, 16, 32, 3, 25088, 1, 1, stream=stream0)
        buf1722 = reinterpret_tensor(buf1699, (512, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1699  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_99.run(buf1711, buf1722, 262144, 49, stream=stream0)
        del buf1711
        # Topologically Sorted Source Nodes: [layer4_2_conv3], Original ATen: [aten.convolution]
        buf1723 = extern_kernels.convolution(buf1722, buf1715, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1723, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        del buf1722
        buf1724 = buf1636; del buf1636  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_111.run(buf1498, buf1724, 2048, stream=stream0)
        buf1725 = reinterpret_tensor(buf1659, (512, 2048, 49), (100352, 49, 1), 0); del buf1659  # reuse
        buf1731 = buf1633; del buf1633  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_102.run(buf1723, buf1725, buf1731, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_25.run(buf1725, buf1498, buf1724, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf1728 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1732 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103.run(buf1724, buf1498, buf1728, buf1732, 2048, stream=stream0)
        buf1729 = buf1725; del buf1725  # reuse
        buf1730 = reinterpret_tensor(buf1723, (512, 2048, 49), (100352, 49, 1), 0); del buf1723  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_26.run(buf1731, buf1732, buf1728, primals_316, primals_317, buf1729, buf1730, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del buf1732
        del primals_317
        buf1735 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1736 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1737 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1730, (100352, 512), (512, 1), 0), buf1735, buf1736, buf1737, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1741 = reinterpret_tensor(buf1730, (512, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1730  # reuse
        buf1742 = reinterpret_tensor(buf1731, (512, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1731  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_105.run(buf1729, buf1645, buf1742, 51380224, stream=stream0)
        del buf1645
        del buf1729
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1742, buf1741, buf1544, 51380224, 1024, 50176, 1, 1, stream=stream0)
        del buf1546
        del buf1646
        del buf1742
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1544, (100352, 512), (512, 1), 0), buf434, 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del buf1544
        buf1747 = empty_strided_cuda((512, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [avgpool, flatten], Original ATen: [aten.mean, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_view_112.run(buf1741, buf1747, 1048576, 49, stream=stream0)
        del buf1741
        buf1748 = empty_strided_cuda((2048, 32), (32, 1), torch.int32)
        buf1749 = empty_strided_cuda((2048, ), (1, ), torch.bfloat16)
        buf1750 = empty_strided_cuda((2048, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1747, (2048, 512), (512, 1), 0), buf1748, buf1749, buf1750, 512, 1, 32, 1, 2, 16, 32, 3, 2048, 1, 1, stream=stream0)
        buf1754 = empty_strided_cuda((100, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [fc], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_113.run(primals_320, buf1754, 204800, stream=stream0)
        del primals_320
        buf1755 = empty_strided_cuda((512, 100), (100, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [fc], Original ATen: [aten.mm]
        extern_kernels.mm(buf1747, reinterpret_tensor(buf1754, (2048, 100), (1, 2048), 0), out=buf1755)
        del buf1747
        # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_3, primals_3, 1, stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_115.run(primals_6, buf11, primals_7, buf12, primals_6, primals_7, 64, stream=stream0)
        del buf11
        del buf12
        del primals_6
        del primals_7
        # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_9, primals_9, 1, stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [layer1_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_116.run(primals_12, buf49, primals_13, buf50, primals_12, primals_13, 64, stream=stream0)
        del buf49
        del buf50
        del primals_12
        del primals_13
        # Topologically Sorted Source Nodes: [add__2], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_15, primals_15, 1, stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_116.run(primals_18, buf84, primals_19, buf85, primals_18, primals_19, 64, stream=stream0)
        del buf84
        del buf85
        del primals_18
        del primals_19
        # Topologically Sorted Source Nodes: [add__3], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_21, primals_21, 1, stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_117.run(primals_24, buf118, primals_25, buf119, primals_24, primals_25, 256, stream=stream0)
        del buf118
        del buf119
        del primals_24
        del primals_25
        # Topologically Sorted Source Nodes: [add__4], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_27, primals_27, 1, stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_downsample_1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_117.run(primals_30, buf145, primals_31, buf146, primals_30, primals_31, 256, stream=stream0)
        del buf145
        del buf146
        del primals_30
        del primals_31
        # Topologically Sorted Source Nodes: [add__5], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_33, primals_33, 1, stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_116.run(primals_36, buf180, primals_37, buf181, primals_36, primals_37, 64, stream=stream0)
        del buf180
        del buf181
        del primals_36
        del primals_37
        # Topologically Sorted Source Nodes: [add__6], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_39, primals_39, 1, stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_116.run(primals_42, buf213, primals_43, buf214, primals_42, primals_43, 64, stream=stream0)
        del buf213
        del buf214
        del primals_42
        del primals_43
        # Topologically Sorted Source Nodes: [add__7], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_45, primals_45, 1, stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_117.run(primals_48, buf246, primals_49, buf247, primals_48, primals_49, 256, stream=stream0)
        del buf246
        del buf247
        del primals_48
        del primals_49
        # Topologically Sorted Source Nodes: [add__8], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_51, primals_51, 1, stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_116.run(primals_54, buf280, primals_55, buf281, primals_54, primals_55, 64, stream=stream0)
        del buf280
        del buf281
        del primals_54
        del primals_55
        # Topologically Sorted Source Nodes: [add__9], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_57, primals_57, 1, stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_116.run(primals_60, buf10, primals_61, buf313, primals_60, primals_61, 64, stream=stream0)
        del buf10
        del buf313
        del primals_60
        del primals_61
        # Topologically Sorted Source Nodes: [add__10], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_63, primals_63, 1, stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_117.run(primals_66, buf344, primals_67, buf345, primals_66, primals_67, 256, stream=stream0)
        del buf344
        del buf345
        del primals_66
        del primals_67
        # Topologically Sorted Source Nodes: [add__11], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_69, primals_69, 1, stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer2_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_118.run(primals_72, buf377, primals_73, buf378, primals_72, primals_73, 128, stream=stream0)
        del buf377
        del buf378
        del primals_72
        del primals_73
        # Topologically Sorted Source Nodes: [add__12], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_75, primals_75, 1, stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [layer2_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_119.run(primals_78, buf411, primals_79, buf412, primals_78, primals_79, 128, stream=stream0)
        del buf411
        del buf412
        del primals_78
        del primals_79
        # Topologically Sorted Source Nodes: [add__13], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_81, primals_81, 1, stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_120.run(primals_84, buf447, primals_85, buf448, primals_84, primals_85, 512, stream=stream0)
        del buf447
        del buf448
        del primals_84
        del primals_85
        # Topologically Sorted Source Nodes: [add__14], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_87, primals_87, 1, stream=stream0)
        del primals_87
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_downsample_1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_120.run(primals_90, buf474, primals_91, buf475, primals_90, primals_91, 512, stream=stream0)
        del buf474
        del buf475
        del primals_90
        del primals_91
        # Topologically Sorted Source Nodes: [add__15], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_93, primals_93, 1, stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_119.run(primals_96, buf509, primals_97, buf510, primals_96, primals_97, 128, stream=stream0)
        del buf509
        del buf510
        del primals_96
        del primals_97
        # Topologically Sorted Source Nodes: [add__16], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_99, primals_99, 1, stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_119.run(primals_102, buf542, primals_103, buf543, primals_102, primals_103, 128, stream=stream0)
        del buf542
        del buf543
        del primals_102
        del primals_103
        # Topologically Sorted Source Nodes: [add__17], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_105, primals_105, 1, stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_120.run(primals_108, buf575, primals_109, buf576, primals_108, primals_109, 512, stream=stream0)
        del buf575
        del buf576
        del primals_108
        del primals_109
        # Topologically Sorted Source Nodes: [add__18], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_111, primals_111, 1, stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_119.run(primals_114, buf609, primals_115, buf610, primals_114, primals_115, 128, stream=stream0)
        del buf609
        del buf610
        del primals_114
        del primals_115
        # Topologically Sorted Source Nodes: [add__19], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_117, primals_117, 1, stream=stream0)
        del primals_117
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_119.run(primals_120, buf642, primals_121, buf643, primals_120, primals_121, 128, stream=stream0)
        del buf642
        del buf643
        del primals_120
        del primals_121
        # Topologically Sorted Source Nodes: [add__20], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_123, primals_123, 1, stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_120.run(primals_126, buf675, primals_127, buf676, primals_126, primals_127, 512, stream=stream0)
        del buf675
        del buf676
        del primals_126
        del primals_127
        # Topologically Sorted Source Nodes: [add__21], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_129, primals_129, 1, stream=stream0)
        del primals_129
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_119.run(primals_132, buf709, primals_133, buf710, primals_132, primals_133, 128, stream=stream0)
        del buf709
        del buf710
        del primals_132
        del primals_133
        # Topologically Sorted Source Nodes: [add__22], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_135, primals_135, 1, stream=stream0)
        del primals_135
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_119.run(primals_138, buf376, primals_139, buf742, primals_138, primals_139, 128, stream=stream0)
        del buf376
        del buf742
        del primals_138
        del primals_139
        # Topologically Sorted Source Nodes: [add__23], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_141, primals_141, 1, stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_120.run(primals_144, buf773, primals_145, buf774, primals_144, primals_145, 512, stream=stream0)
        del buf773
        del buf774
        del primals_144
        del primals_145
        # Topologically Sorted Source Nodes: [add__24], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_147, primals_147, 1, stream=stream0)
        del primals_147
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer3_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_121.run(primals_150, buf805, primals_151, buf806, primals_150, primals_151, 256, stream=stream0)
        del buf805
        del buf806
        del primals_150
        del primals_151
        # Topologically Sorted Source Nodes: [add__25], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_153, primals_153, 1, stream=stream0)
        del primals_153
        # Topologically Sorted Source Nodes: [layer3_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_156, buf838, primals_157, buf839, primals_156, primals_157, 256, stream=stream0)
        del buf838
        del buf839
        del primals_156
        del primals_157
        # Topologically Sorted Source Nodes: [add__26], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_159, primals_159, 1, stream=stream0)
        del primals_159
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_123.run(primals_162, buf874, primals_163, buf875, primals_162, primals_163, 1024, stream=stream0)
        del buf874
        del buf875
        del primals_162
        del primals_163
        # Topologically Sorted Source Nodes: [add__27], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_165, primals_165, 1, stream=stream0)
        del primals_165
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_downsample_1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_123.run(primals_168, buf901, primals_169, buf902, primals_168, primals_169, 1024, stream=stream0)
        del buf901
        del buf902
        del primals_168
        del primals_169
        # Topologically Sorted Source Nodes: [add__28], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_171, primals_171, 1, stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_174, buf936, primals_175, buf937, primals_174, primals_175, 256, stream=stream0)
        del buf936
        del buf937
        del primals_174
        del primals_175
        # Topologically Sorted Source Nodes: [add__29], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_177, primals_177, 1, stream=stream0)
        del primals_177
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_180, buf969, primals_181, buf970, primals_180, primals_181, 256, stream=stream0)
        del buf969
        del buf970
        del primals_180
        del primals_181
        # Topologically Sorted Source Nodes: [add__30], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_183, primals_183, 1, stream=stream0)
        del primals_183
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_123.run(primals_186, buf1002, primals_187, buf1003, primals_186, primals_187, 1024, stream=stream0)
        del buf1002
        del buf1003
        del primals_186
        del primals_187
        # Topologically Sorted Source Nodes: [add__31], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_189, primals_189, 1, stream=stream0)
        del primals_189
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_192, buf1036, primals_193, buf1037, primals_192, primals_193, 256, stream=stream0)
        del buf1036
        del buf1037
        del primals_192
        del primals_193
        # Topologically Sorted Source Nodes: [add__32], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_195, primals_195, 1, stream=stream0)
        del primals_195
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_198, buf1069, primals_199, buf1070, primals_198, primals_199, 256, stream=stream0)
        del buf1069
        del buf1070
        del primals_198
        del primals_199
        # Topologically Sorted Source Nodes: [add__33], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_201, primals_201, 1, stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_123.run(primals_204, buf1102, primals_205, buf1103, primals_204, primals_205, 1024, stream=stream0)
        del buf1102
        del buf1103
        del primals_204
        del primals_205
        # Topologically Sorted Source Nodes: [add__34], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_207, primals_207, 1, stream=stream0)
        del primals_207
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_210, buf1136, primals_211, buf1137, primals_210, primals_211, 256, stream=stream0)
        del buf1136
        del buf1137
        del primals_210
        del primals_211
        # Topologically Sorted Source Nodes: [add__35], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_213, primals_213, 1, stream=stream0)
        del primals_213
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_216, buf1169, primals_217, buf1170, primals_216, primals_217, 256, stream=stream0)
        del buf1169
        del buf1170
        del primals_216
        del primals_217
        # Topologically Sorted Source Nodes: [add__36], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_219, primals_219, 1, stream=stream0)
        del primals_219
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_123.run(primals_222, buf1202, primals_223, buf1203, primals_222, primals_223, 1024, stream=stream0)
        del buf1202
        del buf1203
        del primals_222
        del primals_223
        # Topologically Sorted Source Nodes: [add__37], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_225, primals_225, 1, stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_228, buf1236, primals_229, buf1237, primals_228, primals_229, 256, stream=stream0)
        del buf1236
        del buf1237
        del primals_228
        del primals_229
        # Topologically Sorted Source Nodes: [add__38], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_231, primals_231, 1, stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_234, buf1269, primals_235, buf1270, primals_234, primals_235, 256, stream=stream0)
        del buf1269
        del buf1270
        del primals_234
        del primals_235
        # Topologically Sorted Source Nodes: [add__39], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_237, primals_237, 1, stream=stream0)
        del primals_237
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_123.run(primals_240, buf1302, primals_241, buf1303, primals_240, primals_241, 1024, stream=stream0)
        del buf1302
        del buf1303
        del primals_240
        del primals_241
        # Topologically Sorted Source Nodes: [add__40], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_243, primals_243, 1, stream=stream0)
        del primals_243
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_246, buf1336, primals_247, buf1337, primals_246, primals_247, 256, stream=stream0)
        del buf1336
        del buf1337
        del primals_246
        del primals_247
        # Topologically Sorted Source Nodes: [add__41], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_249, primals_249, 1, stream=stream0)
        del primals_249
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_122.run(primals_252, buf117, primals_253, buf1369, primals_252, primals_253, 256, stream=stream0)
        del buf117
        del buf1369
        del primals_252
        del primals_253
        # Topologically Sorted Source Nodes: [add__42], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_255, primals_255, 1, stream=stream0)
        del primals_255
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_123.run(primals_258, buf873, primals_259, buf1399, primals_258, primals_259, 1024, stream=stream0)
        del buf1399
        del buf873
        del primals_258
        del primals_259
        # Topologically Sorted Source Nodes: [add__43], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_261, primals_261, 1, stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer4_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_124.run(primals_264, buf1430, primals_265, buf1431, primals_264, primals_265, 512, stream=stream0)
        del buf1430
        del buf1431
        del primals_264
        del primals_265
        # Topologically Sorted Source Nodes: [add__44], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_267, primals_267, 1, stream=stream0)
        del primals_267
        # Topologically Sorted Source Nodes: [layer4_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_125.run(primals_270, buf1463, primals_271, buf1464, primals_270, primals_271, 512, stream=stream0)
        del buf1463
        del buf1464
        del primals_270
        del primals_271
        # Topologically Sorted Source Nodes: [add__45], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_273, primals_273, 1, stream=stream0)
        del primals_273
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_126.run(primals_276, buf1499, primals_277, buf1500, primals_276, primals_277, 2048, stream=stream0)
        del buf1499
        del buf1500
        del primals_276
        del primals_277
        # Topologically Sorted Source Nodes: [add__46], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_279, primals_279, 1, stream=stream0)
        del primals_279
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_downsample_1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_126.run(primals_282, buf1526, primals_283, buf1527, primals_282, primals_283, 2048, stream=stream0)
        del buf1526
        del buf1527
        del primals_282
        del primals_283
        # Topologically Sorted Source Nodes: [add__47], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_285, primals_285, 1, stream=stream0)
        del primals_285
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_125.run(primals_288, buf1561, primals_289, buf1562, primals_288, primals_289, 512, stream=stream0)
        del buf1561
        del buf1562
        del primals_288
        del primals_289
        # Topologically Sorted Source Nodes: [add__48], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_291, primals_291, 1, stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_125.run(primals_294, buf1594, primals_295, buf1595, primals_294, primals_295, 512, stream=stream0)
        del buf1594
        del buf1595
        del primals_294
        del primals_295
        # Topologically Sorted Source Nodes: [add__49], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_297, primals_297, 1, stream=stream0)
        del primals_297
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_126.run(primals_300, buf1627, primals_301, buf1628, primals_300, primals_301, 2048, stream=stream0)
        del buf1627
        del buf1628
        del primals_300
        del primals_301
        # Topologically Sorted Source Nodes: [add__50], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_303, primals_303, 1, stream=stream0)
        del primals_303
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_125.run(primals_306, buf1661, primals_307, buf1662, primals_306, primals_307, 512, stream=stream0)
        del buf1661
        del buf1662
        del primals_306
        del primals_307
        # Topologically Sorted Source Nodes: [add__51], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_309, primals_309, 1, stream=stream0)
        del primals_309
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_125.run(primals_312, buf446, primals_313, buf1694, primals_312, primals_313, 512, stream=stream0)
        del buf1694
        del buf446
        del primals_312
        del primals_313
        # Topologically Sorted Source Nodes: [add__52], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__114.run(primals_315, primals_315, 1, stream=stream0)
        del primals_315
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_126.run(primals_318, buf1498, primals_319, buf1724, primals_318, primals_319, 2048, stream=stream0)
        del buf1498
        del buf1724
        del primals_318
        del primals_319
    return (buf1755, primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, buf0, buf1, buf2, buf16, buf23, buf24, buf25, buf31, reinterpret_tensor(buf36, (802816, 16), (16, 1), 0), buf39, buf40, buf41, buf42, buf43, buf54, buf61, buf62, buf63, reinterpret_tensor(buf73, (200704, 16), (16, 1), 0), buf75, buf76, buf77, buf78, buf89, buf96, buf97, buf98, reinterpret_tensor(buf106, (200704, 16), (16, 1), 0), buf108, buf109, buf110, buf111, buf123, buf130, buf131, buf132, buf136, buf137, buf138, buf139, buf150, buf157, buf158, buf159, reinterpret_tensor(buf169, (802816, 16), (16, 1), 0), buf171, buf172, buf173, buf174, buf185, buf192, buf193, buf194, reinterpret_tensor(buf202, (200704, 16), (16, 1), 0), buf204, buf205, buf206, buf207, buf218, buf225, buf226, buf227, reinterpret_tensor(buf235, (200704, 16), (16, 1), 0), buf237, buf238, buf239, buf240, buf251, buf258, buf259, buf260, reinterpret_tensor(buf269, (802816, 16), (16, 1), 0), buf271, buf272, buf273, buf274, buf285, buf292, buf293, buf294, reinterpret_tensor(buf302, (200704, 16), (16, 1), 0), buf304, buf305, buf306, buf307, buf317, buf324, buf325, buf326, reinterpret_tensor(buf333, (200704, 16), (16, 1), 0), buf335, buf336, buf337, buf338, buf349, buf356, buf357, buf358, buf35, buf367, buf368, buf369, buf370, buf382, buf389, buf390, buf391, reinterpret_tensor(buf400, (401408, 16), (16, 1), 0), buf402, buf403, buf404, buf405, buf416, buf423, buf424, buf425, reinterpret_tensor(buf435, (100352, 16), (16, 1), 0), buf437, buf438, buf439, buf440, buf452, buf459, buf460, buf461, buf465, buf466, buf467, buf468, buf479, buf486, buf487, buf488, reinterpret_tensor(buf498, (401408, 16), (16, 1), 0), buf500, buf501, buf502, buf503, buf514, buf521, buf522, buf523, reinterpret_tensor(buf531, (100352, 16), (16, 1), 0), buf533, buf534, buf535, buf536, buf547, buf554, buf555, buf556, reinterpret_tensor(buf564, (100352, 16), (16, 1), 0), buf566, buf567, buf568, buf569, buf580, buf587, buf588, buf589, reinterpret_tensor(buf598, (401408, 16), (16, 1), 0), buf600, buf601, buf602, buf603, buf614, buf621, buf622, buf623, reinterpret_tensor(buf631, (100352, 16), (16, 1), 0), buf633, buf634, buf635, buf636, buf647, buf654, buf655, buf656, reinterpret_tensor(buf664, (100352, 16), (16, 1), 0), buf666, buf667, buf668, buf669, buf680, buf687, buf688, buf689, reinterpret_tensor(buf698, (401408, 16), (16, 1), 0), buf700, buf701, buf702, buf703, buf714, buf721, buf722, buf723, reinterpret_tensor(buf731, (100352, 16), (16, 1), 0), buf733, buf734, buf735, buf736, buf746, buf753, buf754, buf755, reinterpret_tensor(buf762, (100352, 16), (16, 1), 0), buf764, buf765, buf766, buf767, buf778, buf785, buf786, buf787, buf399, buf796, buf797, buf798, buf799, buf810, buf817, buf818, buf819, reinterpret_tensor(buf827, (200704, 16), (16, 1), 0), buf829, buf830, buf831, buf832, buf843, buf850, buf851, buf852, reinterpret_tensor(buf862, (50176, 16), (16, 1), 0), buf864, buf865, buf866, buf867, buf879, buf886, buf887, buf888, buf892, buf893, buf894, buf895, buf906, buf913, buf914, buf915, reinterpret_tensor(buf925, (200704, 16), (16, 1), 0), buf927, buf928, buf929, buf930, buf941, buf948, buf949, buf950, reinterpret_tensor(buf958, (50176, 16), (16, 1), 0), buf960, buf961, buf962, buf963, buf974, buf981, buf982, buf983, reinterpret_tensor(buf991, (50176, 16), (16, 1), 0), buf993, buf994, buf995, buf996, buf1007, buf1014, buf1015, buf1016, reinterpret_tensor(buf1025, (200704, 16), (16, 1), 0), buf1027, buf1028, buf1029, buf1030, buf1041, buf1048, buf1049, buf1050, reinterpret_tensor(buf1058, (50176, 16), (16, 1), 0), buf1060, buf1061, buf1062, buf1063, buf1074, buf1081, buf1082, buf1083, reinterpret_tensor(buf1091, (50176, 16), (16, 1), 0), buf1093, buf1094, buf1095, buf1096, buf1107, buf1114, buf1115, buf1116, reinterpret_tensor(buf1125, (200704, 16), (16, 1), 0), buf1127, buf1128, buf1129, buf1130, buf1141, buf1148, buf1149, buf1150, reinterpret_tensor(buf1158, (50176, 16), (16, 1), 0), buf1160, buf1161, buf1162, buf1163, buf1174, buf1181, buf1182, buf1183, reinterpret_tensor(buf1191, (50176, 16), (16, 1), 0), buf1193, buf1194, buf1195, buf1196, buf1207, buf1214, buf1215, buf1216, reinterpret_tensor(buf1225, (200704, 16), (16, 1), 0), buf1227, buf1228, buf1229, buf1230, buf1241, buf1248, buf1249, buf1250, reinterpret_tensor(buf1258, (50176, 16), (16, 1), 0), buf1260, buf1261, buf1262, buf1263, buf1274, buf1281, buf1282, buf1283, reinterpret_tensor(buf1291, (50176, 16), (16, 1), 0), buf1293, buf1294, buf1295, buf1296, buf1307, buf1314, buf1315, buf1316, reinterpret_tensor(buf1325, (200704, 16), (16, 1), 0), buf1327, buf1328, buf1329, buf1330, buf1341, buf1348, buf1349, buf1350, reinterpret_tensor(buf1358, (50176, 16), (16, 1), 0), buf1360, buf1361, buf1362, buf1363, buf1373, buf1380, buf1381, buf1382, buf861, buf1390, buf1391, buf1392, buf1393, buf1403, buf1410, buf1411, buf1412, buf72, buf1421, buf1422, buf1423, buf1424, buf1435, buf1442, buf1443, buf1444, reinterpret_tensor(buf1452, (100352, 16), (16, 1), 0), buf1454, buf1455, buf1456, buf1457, buf1468, buf1475, buf1476, buf1477, reinterpret_tensor(buf1487, (25088, 16), (16, 1), 0), buf1489, buf1490, buf1491, buf1492, buf1504, buf1511, buf1512, buf1513, buf1517, buf1518, buf1519, buf1520, buf1531, buf1538, buf1539, buf1540, reinterpret_tensor(buf1550, (100352, 16), (16, 1), 0), buf1552, buf1553, buf1554, buf1555, buf1566, buf1573, buf1574, buf1575, reinterpret_tensor(buf1583, (25088, 16), (16, 1), 0), buf1585, buf1586, buf1587, buf1588, buf1599, buf1606, buf1607, buf1608, reinterpret_tensor(buf1616, (25088, 16), (16, 1), 0), buf1618, buf1619, buf1620, buf1621, buf1632, buf1639, buf1640, buf1641, reinterpret_tensor(buf1650, (100352, 16), (16, 1), 0), buf1652, buf1653, buf1654, buf1655, buf1666, buf1673, buf1674, buf1675, reinterpret_tensor(buf1683, (25088, 16), (16, 1), 0), buf1685, buf1686, buf1687, buf1688, buf1698, buf1705, buf1706, buf1707, buf1486, buf1715, buf1716, buf1717, buf1718, buf1728, buf1735, buf1736, buf1737, buf434, buf1748, buf1749, buf1750, buf1754, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_52 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_118 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_148 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_160 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_166 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_178 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_184 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_190 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_202 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_238 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_256 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_262 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_274 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_280 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_316 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((100, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
