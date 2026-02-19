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


# kernel path: /tmp/torchinductor_yyu496/mz/cmz54iyxjhppbfk26trp6filbx575cqlumrxeehdiuv2whrlwehw.py
# Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy, aten.avg_pool2d]
# Source node to ATen node mapping:
#   conv1 => avg_pool2d, convert_element_type, convert_element_type_2
# Graph fragment:
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%convert_element_type, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_0 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 8192}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67289088, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_0(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 5476
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 74)
    x2 = xindex // 74
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 3)
    y4 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (3*x1 + 672*x2 + 50176*y0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 3*x1 + 672*x2 + 50176*y0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 3*x1 + 672*x2 + 50176*y0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (224 + 3*x1 + 672*x2 + 50176*y0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (225 + 3*x1 + 672*x2 + 50176*y0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (226 + 3*x1 + 672*x2 + 50176*y0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (448 + 3*x1 + 672*x2 + 50176*y0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (449 + 3*x1 + 672*x2 + 50176*y0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (450 + 3*x1 + 672*x2 + 50176*y0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 + tmp4
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 + tmp7
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 + tmp10
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 + tmp13
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 + tmp16
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 + tmp19
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 + tmp22
    tmp26 = 0.1111111111111111
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 3*x5 + 16428*y4), tmp28, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3f/c3fhyuujely6vicn4lztyh6g666ushjasfnx7npqdu2bzhp2erqu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_310
# Graph fragment:
#   %triton_kernel_wrapper_mutation_310 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 218, constant_args_idx: 311, grid: [(602112, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %view_1, P_ptr: %empty, S_ptr: %empty_1, M_ptr: %empty_2, stride_x0: 512, stride_x1: 1, stride_p0: 32, stride_p1: 1, BITS: 2, VPW: 16, NWORDS: 32, QMAX: 3}})
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2466250752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 308281344
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


# kernel path: /tmp/torchinductor_yyu496/5k/c5k6pgh62edehjaqpc34avjqpxqssvvwyiocm667ps2mjqmj5ian.py
# Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   conv1 => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 65536}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1233125376, 'x': 1233125376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/dg/cdgtnaaaaf6o2p6ibfakfpzj7tmfprq5jzdrorprdtrlkmqsylf4.py
# Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   conv1 => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 37632, 'x': 37632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/uc/cuch5lnznsbzehi75yo2nmpyjpuben242372vihv57muryffloz2.py
# Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   bn1 => full_default
# Graph fragment:
#   %full_default : [num_users=15] = call_function[target=torch.ops.aten.full.default](args = ([64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_4 = async_compile.triton('triton_poi_fused_zeros_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ww/cww4ym46y26mfwiv3sh3r5n7n7x57ugujhpt7ylryd346ksswgds.py
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
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3328}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/52/c52woovpbg6vatxhrttsuekiwcp4fl43gdcqz3kc2j2aq5gjteln.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_308, triton_kernel_wrapper_mutation_309
# Graph fragment:
#   %triton_kernel_wrapper_mutation_309 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 219, constant_args_idx: 312, grid: [(64, 25088, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_7, SUM: %as_strided_default_357, SUMSQ: %as_strided_default_359, M: 25690112, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_308 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 220, constant_args_idx: 313, grid: [(64, 25088, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_7, MEAN: %div, INVSTD: %rsqrt, GAMMA: %primals_4, BETA: %primals_5, Y: %permute, X_hat: %permute_1, M: 25690112, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3288334336, 'x': 13153337344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 12544
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 802816*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 12544*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 12544*y3), tmp0, xmask & ymask)
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


# kernel path: /tmp/torchinductor_yyu496/a3/ca3bpo4aadp3smatt4a4ku6mzbryshchstorowqh6g4z2vthjoqx.py
# Topologically Sorted Source Nodes: [bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_308
#   bn1 => add_1, clamp_min, div, div_1, full_default_2, mul_1, rsqrt, sub
# Graph fragment:
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 25690112.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_357, %full_default_2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_359, %full_default_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %div), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %mul_1), kwargs = {})
#   %clamp_min : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %triton_kernel_wrapper_mutation_308 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 220, constant_args_idx: 313, grid: [(64, 25088, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_7, MEAN: %div, INVSTD: %rsqrt, GAMMA: %primals_4, BETA: %primals_5, Y: %permute, X_hat: %permute_1, M: 25690112, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
triton_poi_fused_add_div_mul_rsqrt_sub_7 = async_compile.triton('triton_poi_fused_add_div_mul_rsqrt_sub_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_rsqrt_sub_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_rsqrt_sub_7(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = 3.8925482302295915e-08
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


# kernel path: /tmp/torchinductor_yyu496/hi/chid5fhw7gg2jzw6zgyex2jas7pjtwillpeuqj3lgxkvxmlhdwxd.py
# Topologically Sorted Source Nodes: [relu, ], Original ATen: [aten.empty_like]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_306
#   relu => permute_2
# Graph fragment:
#   %permute_2 : [num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%empty_8, [0, 1, 2, 3]), kwargs = {})
#   %triton_kernel_wrapper_mutation_306 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 315, grid: [(1605632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %view_18, Y_ptr: %permute_2, Mask_prt: %full_default_4, n_elts: 1644167168, BLOCK_SIZE: 1024}})
triton_poi_fused_empty_like_8 = async_compile.triton('triton_poi_fused_empty_like_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_empty_like_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9865003008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_empty_like_8(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
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


# kernel path: /tmp/torchinductor_yyu496/34/c34fpxhm4a3wmgcudorevnrcckt46s7ryugzadfzgbyjp3btyjib.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_306
# Graph fragment:
#   %triton_kernel_wrapper_mutation_306 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 315, grid: [(1605632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %view_18, Y_ptr: %permute_2, Mask_prt: %full_default_4, n_elts: 1644167168, BLOCK_SIZE: 1024}})
triton_poi_fused_9 = async_compile.triton('triton_poi_fused_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
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


# kernel path: /tmp/torchinductor_yyu496/e6/ce6b6wekarn4autjv7ptrigp33d4lkhelxckpfg3x7ottco4houa.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   relu => full_default_5
# Graph fragment:
#   %full_default_5 : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([3211264, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_10 = async_compile.triton('triton_poi_fused_zeros_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xx/cxxfsfhghifaljk6evekts2rsdlefzuzjvbb3a3rrefrscm4ef5o.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_177
# Graph fragment:
#   %clone_default_177 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_354,), kwargs = {})
triton_poi_fused_11 = async_compile.triton('triton_poi_fused_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 616562688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
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


# kernel path: /tmp/torchinductor_yyu496/mw/cmwbp44eixzdgtil5nao2ffw2twdhmv5nbec4gyle63srrurrmf7.py
# Topologically Sorted Source Nodes: [maxpool, layer1_0_conv1, layer1_0_downsample_0], Original ATen: [aten.max_pool2d_with_indices, aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_conv1 => convolution_1
#   layer1_0_downsample_0 => convolution_4
#   maxpool => _low_memory_max_pool_with_offsets, getitem_14
# Graph fragment:
#   %_low_memory_max_pool_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%permute_2, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_14 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 1), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_13, %convert_element_type_3, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_13, %convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_12 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i8', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4110417920, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_12(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 3136
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
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
    tmp11 = tl.load(in_ptr0 + ((-113) + 2*x1 + 224*x2 + 12544*y0), tmp10 & xmask & ymask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-112) + 2*x1 + 224*x2 + 12544*y0), tmp16 & xmask & ymask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-111) + 2*x1 + 224*x2 + 12544*y0), tmp23 & xmask & ymask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x1 + 224*x2 + 12544*y0), tmp30 & xmask & ymask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x1 + 224*x2 + 12544*y0), tmp33 & xmask & ymask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x1 + 224*x2 + 12544*y0), tmp36 & xmask & ymask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (111 + 2*x1 + 224*x2 + 12544*y0), tmp43 & xmask & ymask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (112 + 2*x1 + 224*x2 + 12544*y0), tmp46 & xmask & ymask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (113 + 2*x1 + 224*x2 + 12544*y0), tmp49 & xmask & ymask, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
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
    tl.store(out_ptr0 + (x5 + 3136*y0), tmp51, xmask & ymask)
    tl.store(out_ptr1 + (y3 + 64*x5 + 200704*y4), tmp164, xmask & ymask)
    tl.store(out_ptr2 + (y3 + 64*x5 + 200704*y4), tmp51, xmask & ymask)
    tl.store(out_ptr3 + (y3 + 64*x5 + 200704*y4), tmp51, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pr/cprpwuqiw7x6wk2o2w4adu3a6v4j7sp7xkmu4yzpja4f3d6cjgc4.py
# Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer1_0_conv1 => convert_element_type_3
# Graph fragment:
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_13 = async_compile.triton('triton_poi_fused__to_copy_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/py/cpyd4n6dc2inmf626qvbugqxb62elh4bbzffis2gv2qfmbelaz3z.py
# Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer1_0_conv1 => avg_pool2d_1, convert_element_type_4
# Graph fragment:
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_13, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_1, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_14 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 84934656, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_14(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 324
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 18)
    x2 = xindex // 18
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 64)
    y4 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (56 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (57 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (58 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (112 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (113 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (114 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 64*x5 + 20736*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bg/cbgn4we3lrmdalmrpwdnqtayjq6q7dcnc7kfw3ewapsqi4rhylcc.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_302, triton_kernel_wrapper_mutation_303
# Graph fragment:
#   %triton_kernel_wrapper_mutation_303 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 223, constant_args_idx: 318, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_30, SUM: %as_strided_default_351, SUMSQ: %as_strided_default_353, M: 6422528, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_302 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 224, constant_args_idx: 319, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_30, MEAN: %div_3, INVSTD: %rsqrt_1, GAMMA: %primals_10, BETA: %primals_11, Y: %permute_3, X_hat: %permute_4, M: 6422528, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_15 = async_compile.triton('triton_poi_fused_15', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_15(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 3136
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp0, xmask & ymask)
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


# kernel path: /tmp/torchinductor_yyu496/kw/ckww57zb5e7hkv2se57t3cmkwanku6whxzo7s6zviw6x4zkmkoum.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_302
#   layer1_0_bn1 => add_5, clamp_min_2, div_3, div_4, full_default_8, mul_8, rsqrt_1, sub_2
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 6422528.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_3 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_351, %full_default_8), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_353, %full_default_8), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %div_3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_4, %mul_8), kwargs = {})
#   %clamp_min_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %triton_kernel_wrapper_mutation_302 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 224, constant_args_idx: 319, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_30, MEAN: %div_3, INVSTD: %rsqrt_1, GAMMA: %primals_10, BETA: %primals_11, Y: %permute_3, X_hat: %permute_4, M: 6422528, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/vr/cvrs65mficnvkjhmcovtz47cmnk336mj43ravev57gptzeu2nens.py
# Topologically Sorted Source Nodes: [layer1_0_relu], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer1_0_relu => full_default_10
# Graph fragment:
#   %full_default_10 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([2048, 64, 56, 56], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_17 = async_compile.triton('triton_poi_fused_zeros_17', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_17(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/w5/cw5ch7eahwis5kknqyylcfz5x3x4nyyt6gc6a7hotxykz4n4v6yl.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_170, clone_default_174
# Graph fragment:
#   %clone_default_174 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_348,), kwargs = {})
#   %clone_default_170 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_340,), kwargs = {})
triton_poi_fused_18 = async_compile.triton('triton_poi_fused_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2055208960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_18(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2w/c2wfznij363nfsgyu6sseatncokogibcgwjqf2cka6lfxcdfn6xx.py
# Topologically Sorted Source Nodes: [layer1_0_relu], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer1_0_relu => full_default_11
# Graph fragment:
#   %full_default_11 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([802816, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_19 = async_compile.triton('triton_poi_fused_zeros_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_19(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/54/c54mwjrthc54i7rf2vusmruznsv7tfcaa5nc6neftut67mqo5acu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_169, clone_default_173
# Graph fragment:
#   %clone_default_173 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_346,), kwargs = {})
#   %clone_default_169 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_338,), kwargs = {})
triton_poi_fused_20 = async_compile.triton('triton_poi_fused_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'out_ptr1': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256901120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_20(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yv/cyvkiveg47cpmkgvm64fde5h3g4cpstwsakrnpeimp324udjqkm2.py
# Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer1_0_conv2 => convert_element_type_5
# Graph fragment:
#   %convert_element_type_5 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_14, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_21 = async_compile.triton('triton_poi_fused__to_copy_21', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 147456, 'x': 147456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/zp/czp6u7n53erpkrqhwn3eoguroslafbw2szeuz6l7uo5wfoi3tqmd.py
# Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_conv2 => convolution_2
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_5, %convert_element_type_5, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_22 = async_compile.triton('triton_poi_fused_convolution_22', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 64*x2 + 200704*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4m/c4myjoclxxyai5cxl27433l5bzkter43gmef76tdgby2v66mhuce.py
# Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer1_0_conv3 => convert_element_type_7
# Graph fragment:
#   %convert_element_type_7 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_20, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_23 = async_compile.triton('triton_poi_fused__to_copy_23', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ql/cqlwqwfbh57xshpnyx4j22c25fmkxd6y7mtsrrcrehz6yrmcmy6w.py
# Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer1_0_bn3 => full_default_18
# Graph fragment:
#   %full_default_18 : [num_users=33] = call_function[target=torch.ops.aten.full.default](args = ([256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_24 = async_compile.triton('triton_poi_fused_zeros_24', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_24(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/n6/cn6sun3kpwmwvzjlzirfgjapgah6st3h6ojgep2nmd2typzcpzbv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_165, clone_default_166, clone_default_167, clone_default_168
# Graph fragment:
#   %clone_default_167 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_334,), kwargs = {})
#   %clone_default_168 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_336,), kwargs = {})
#   %clone_default_165 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_330,), kwargs = {})
#   %clone_default_166 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_332,), kwargs = {})
triton_poi_fused_25 = async_compile.triton('triton_poi_fused_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_25(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/eq/ceqvdlcd45xjhhjqopgmaspzir6g2pauoci3zi2qvh772fsco5ed.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_290, triton_kernel_wrapper_mutation_291
# Graph fragment:
#   %triton_kernel_wrapper_mutation_291 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 231, constant_args_idx: 330, grid: [(256, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_80, SUM: %as_strided_default_335, SUMSQ: %as_strided_default_337, M: 6422528, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_290 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 232, constant_args_idx: 331, grid: [(256, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_80, MEAN: %div_9, INVSTD: %rsqrt_3, GAMMA: %primals_22, BETA: %primals_23, Y: %permute_9, X_hat: %permute_10, M: 6422528, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_26 = async_compile.triton('triton_poi_fused_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3288334336, 'x': 13153337344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_26(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
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


# kernel path: /tmp/torchinductor_yyu496/ue/cuexyqrg746sfoxrqr5z6gwmm63kjmv6dp5izow2zd5xf6k6adhz.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_290
#   layer1_0_bn1 => full_default_8
#   layer1_0_bn3 => add_13, clamp_min_6, div_10, div_9, mul_22, rsqrt_3, sub_6
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 6422528.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_9 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_335, %full_default_8), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_337, %full_default_8), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %div_9), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_10, %mul_22), kwargs = {})
#   %clamp_min_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_6, 0.0), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %triton_kernel_wrapper_mutation_290 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 232, constant_args_idx: 331, grid: [(256, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_80, MEAN: %div_9, INVSTD: %rsqrt_3, GAMMA: %primals_22, BETA: %primals_23, Y: %permute_9, X_hat: %permute_10, M: 6422528, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_27 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_27', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_27(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_yyu496/yk/cyk4gw32sibbplaq3a7tjj3k4x2vbvbmzi55ow342u46pmmsgdzo.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_164
# Graph fragment:
#   %clone_default_164 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_328,), kwargs = {})
triton_poi_fused_28 = async_compile.triton('triton_poi_fused_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4932501504}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jm/cjm4yvm5p6gkyb276dzke27actfh2bnpvtuo73admxgwdez7nujr.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_284
# Graph fragment:
#   %triton_kernel_wrapper_mutation_284 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 337, grid: [(1605632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %add_20, Y_ptr: %permute_13, Mask_prt: %as_strided_default_329, n_elts: 1644167168, BLOCK_SIZE: 1024}})
triton_poi_fused_29 = async_compile.triton('triton_poi_fused_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6576668672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/55/c55tvaooqniu7gn3hbk2h7q66ciydgdytqpkxfjzfmodk2q7vhcy.py
# Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer1_1_conv1 => avg_pool2d_5, convert_element_type_12
# Graph fragment:
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_13, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_5, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_30 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 339738624, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_30(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 324
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 18)
    x2 = xindex // 18
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 256)
    y4 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (56 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (57 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (58 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (112 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (113 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (114 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 256*x5 + 82944*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/56/c56keinhqbfrr5vo36qdk3uon4enp6if6swwzzwkkbqju4xp7dum.py
# Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer1_1_conv1 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_13, %convert_element_type_11, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_31 = async_compile.triton('triton_poi_fused_convolution_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 6576668672, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
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


# kernel path: /tmp/torchinductor_yyu496/5c/c5cpxmefxjz4gkwgzyrq65z4p2sn5fzoyoislrqs5qxie2qyaxoc.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_157, clone_default_158, clone_default_161, clone_default_162
# Graph fragment:
#   %clone_default_161 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_322,), kwargs = {})
#   %clone_default_162 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_324,), kwargs = {})
#   %clone_default_157 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_314,), kwargs = {})
#   %clone_default_158 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_316,), kwargs = {})
triton_poi_fused_32 = async_compile.triton('triton_poi_fused_32', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_32(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/xq/cxqxkzcrjxm6ptqsrm3f6f2bwesujs6mjzbrxt3c6dwgk7emelny.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_153, clone_default_154
# Graph fragment:
#   %clone_default_153 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_306,), kwargs = {})
#   %clone_default_154 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_308,), kwargs = {})
triton_poi_fused_33 = async_compile.triton('triton_poi_fused_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_33(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/73/c73shvgs55lgbkrwepji3ibdjlef4slo5uatefhgtay6ampvddpx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_146, clone_default_149, clone_default_150
# Graph fragment:
#   %clone_default_149 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_298,), kwargs = {})
#   %clone_default_150 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_300,), kwargs = {})
#   %clone_default_146 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_292,), kwargs = {})
triton_poi_fused_34 = async_compile.triton('triton_poi_fused_34', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_34(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/n6/cn6xhdv7voccj76qq27utu5vyig3awiakri5wu6slbqqlhaudilg.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_148
# Graph fragment:
#   %clone_default_148 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_296,), kwargs = {})
triton_poi_fused_35 = async_compile.triton('triton_poi_fused_35', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1233125376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/kt/cktpg5cma5gqswbox6fpjubja4ub5ftpmidqnjwspwuvtt5mmviv.py
# Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_conv1 => convert_element_type_23
# Graph fragment:
#   %convert_element_type_23 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_68, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_36 = async_compile.triton('triton_poi_fused__to_copy_36', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fo/cforekb43albis46iti34emmkeeab2v7rr2ogyg7hdt5yu3ax562.py
# Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_bn1 => full_default_64
# Graph fragment:
#   %full_default_64 : [num_users=17] = call_function[target=torch.ops.aten.full.default](args = ([128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_37 = async_compile.triton('triton_poi_fused_zeros_37', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_37(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xs/cxsdhc7doez2ot3rppfh5hn45jjpvhzgshucgie4tkoa7fanfgdu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_138, clone_default_139, clone_default_141, clone_default_142
# Graph fragment:
#   %clone_default_141 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_282,), kwargs = {})
#   %clone_default_142 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_284,), kwargs = {})
#   %clone_default_138 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_276,), kwargs = {})
#   %clone_default_139 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_278,), kwargs = {})
triton_poi_fused_38 = async_compile.triton('triton_poi_fused_38', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_38(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/ut/cutdvrz2thyvacplpk74mxdhgpmptefefpbuaz4ygfzwl4n27ulm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_244, triton_kernel_wrapper_mutation_245
# Graph fragment:
#   %triton_kernel_wrapper_mutation_245 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 263, constant_args_idx: 376, grid: [(128, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_274, SUM: %as_strided_default_283, SUMSQ: %as_strided_default_285, M: 6422528, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_244 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 264, constant_args_idx: 377, grid: [(128, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_274, MEAN: %div_33, INVSTD: %rsqrt_11, GAMMA: %primals_70, BETA: %primals_71, Y: %permute_32, X_hat: %permute_33, M: 6422528, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_39 = async_compile.triton('triton_poi_fused_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 6576668672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_39(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
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


# kernel path: /tmp/torchinductor_yyu496/lf/clf5satuoovwhhvpwehuqtopsb3caocr7wpuapgldz5uoodxrgdx.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer2_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_244
#   layer1_0_bn1 => full_default_8
#   layer2_0_bn1 => add_48, clamp_min_22, div_33, div_34, mul_78, rsqrt_11, sub_22
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 6422528.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_33 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_283, %full_default_8), kwargs = {})
#   %div_34 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_285, %full_default_8), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_33, %div_33), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_34, %mul_78), kwargs = {})
#   %clamp_min_22 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_22, 0.0), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_22, 1e-05), kwargs = {})
#   %rsqrt_11 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_48,), kwargs = {})
#   %triton_kernel_wrapper_mutation_244 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 264, constant_args_idx: 377, grid: [(128, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_274, MEAN: %div_33, INVSTD: %rsqrt_11, GAMMA: %primals_70, BETA: %primals_71, Y: %permute_32, X_hat: %permute_33, M: 6422528, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_40 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_40', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_40(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_yyu496/e3/ce3ot3jodnsjko6572liwqb2d42jcmv54d36gjpixp4evxfkuwkd.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_242
# Graph fragment:
#   %triton_kernel_wrapper_mutation_242 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 379, grid: [(802816, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %view_285, Y_ptr: %permute_34, Mask_prt: %full_default_68, n_elts: 822083584, BLOCK_SIZE: 1024}})
triton_poi_fused_41 = async_compile.triton('triton_poi_fused_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_41(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 822083584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/cl/cclipeoslh44aa6z2dcv4aapbkng4cj5v7t2jdwswkwqqf4p64qr.py
# Topologically Sorted Source Nodes: [layer2_0_relu], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_relu => full_default_69
# Graph fragment:
#   %full_default_69 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([1605632, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_42 = async_compile.triton('triton_poi_fused_zeros_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_42(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xz/cxztp73jonpjnac5ac7ugm2g7x2ll6lhokb6l2bbzahyke3jiszm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_140
# Graph fragment:
#   %clone_default_140 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_280,), kwargs = {})
triton_poi_fused_43 = async_compile.triton('triton_poi_fused_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 308281344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/z3/cz3n4cctfgfhcoxttoyoni5todoy4zvhzfqxps5ybjihaupmmewv.py
# Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_conv2 => convert_element_type_25
# Graph fragment:
#   %convert_element_type_25 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_74, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_44 = async_compile.triton('triton_poi_fused__to_copy_44', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 589824, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_44(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/kz/ckz47gzjlkjdchal5qhohl3orgpa7qb645caxbowvb2ctsdm7lcx.py
# Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_conv2 => avg_pool2d_12, convert_element_type_26
# Graph fragment:
#   %avg_pool2d_12 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_34, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_26 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_12, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_45 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 169869312, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_45(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 324
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 18)
    x2 = xindex // 18
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 128)
    y4 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (56 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (57 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (58 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (112 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (113 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (114 + 3*x1 + 168*x2 + 3136*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 128*x5 + 41472*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bq/cbqfzza2kj45e5qjfpj5vtjv5ii4opd6bp3uwhh5j37g6hhgwk2k.py
# Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_conv2 => convolution_12
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_34, %convert_element_type_25, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_46 = async_compile.triton('triton_poi_fused_convolution_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3288334336, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_46(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
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


# kernel path: /tmp/torchinductor_yyu496/4j/c4jnw3agqiesmped63czbnscfcanny5ny65vommc2i4wxn7w4jf4.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_238, triton_kernel_wrapper_mutation_239
# Graph fragment:
#   %triton_kernel_wrapper_mutation_239 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 267, constant_args_idx: 382, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_299, SUM: %as_strided_default_277, SUMSQ: %as_strided_default_279, M: 1605632, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_238 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 268, constant_args_idx: 383, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_299, MEAN: %div_36, INVSTD: %rsqrt_12, GAMMA: %primals_76, BETA: %primals_77, Y: %permute_35, X_hat: %permute_36, M: 1605632, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_47 = async_compile.triton('triton_poi_fused_47', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_47(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
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


# kernel path: /tmp/torchinductor_yyu496/mb/cmb3ietdbfmikgi5f7zxptyhwal4zjcxn3tsg3b6vxhjns327js6.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_238
#   layer2_0_bn2 => add_52, clamp_min_24, div_36, div_37, full_default_72, mul_85, rsqrt_12, sub_24
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_36 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_277, %full_default_72), kwargs = {})
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_279, %full_default_72), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_36, %div_36), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_37, %mul_85), kwargs = {})
#   %clamp_min_24 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0.0), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_24, 1e-05), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_52,), kwargs = {})
#   %triton_kernel_wrapper_mutation_238 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 268, constant_args_idx: 383, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_299, MEAN: %div_36, INVSTD: %rsqrt_12, GAMMA: %primals_76, BETA: %primals_77, Y: %permute_35, X_hat: %permute_36, M: 1605632, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/y7/cy7q6szlq7bmyfyqnxx5yvkgqjkklqufgkpo7xlsax2fuworq5ry.py
# Topologically Sorted Source Nodes: [layer2_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_relu_1 => full_default_74
# Graph fragment:
#   %full_default_74 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([2048, 128, 28, 28], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_49 = async_compile.triton('triton_poi_fused_zeros_49', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_49(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/oj/cojn6tchkunkonxiqj7swf3dkqzmsa6bgoodug3ecfdacezqq575.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_137
# Graph fragment:
#   %clone_default_137 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_274,), kwargs = {})
triton_poi_fused_50 = async_compile.triton('triton_poi_fused_50', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 616562688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_50(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/en/cendmivh5jjxul3sqi5dpsbs7ebkbdtsscyef5hrintiiwtraael.py
# Topologically Sorted Source Nodes: [layer2_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_relu_1 => full_default_75
# Graph fragment:
#   %full_default_75 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([401408, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_51 = async_compile.triton('triton_poi_fused_zeros_51', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51380224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_51(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/45/c45tmt5gfqs76njilqxxdiiraxr4wv2p73xqchxyesuk5haa6yaa.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_136
# Graph fragment:
#   %clone_default_136 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_272,), kwargs = {})
triton_poi_fused_52 = async_compile.triton('triton_poi_fused_52', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 77070336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7m/c7minsj5dcjtsojzmoy5uy2x2rvxn4ngg3zp745uuoae3vccqa5n.py
# Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_conv3 => convert_element_type_27
# Graph fragment:
#   %convert_element_type_27 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_80, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_53 = async_compile.triton('triton_poi_fused__to_copy_53', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/rx/crxbmnwbmiogxy36hd742ypconvyyk2bxpjtqrnfhcg6jprni7l5.py
# Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_conv3 => avg_pool2d_13, convert_element_type_28
# Graph fragment:
#   %avg_pool2d_13 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_37, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_28 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_13, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_54 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 42467328, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_54(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 81
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 9)
    x2 = xindex // 9
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 128)
    y4 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (28 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (29 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (30 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (56 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (57 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (58 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 128*x5 + 10368*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pt/cptswueagt5kl274nnwkcgbp7aawlyodqkjlphaesm32aeldgs73.py
# Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_conv3 => convolution_13
# Graph fragment:
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_37, %convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_55 = async_compile.triton('triton_poi_fused_convolution_55', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_55(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x2 + 100352*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/a2/ca2rjp54lvl5uqgz774ekuhbvpa73xvzkkfh42au6fz4dwdatm3x.py
# Topologically Sorted Source Nodes: [layer2_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_bn3 => full_default_76
# Graph fragment:
#   %full_default_76 : [num_users=23] = call_function[target=torch.ops.aten.full.default](args = ([512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_56 = async_compile.triton('triton_poi_fused_zeros_56', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_56(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ka/ckaij3kysgjlg7j4kc27pwzx5fecolyxwny6gd4qkv77uynzyh7b.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_132, clone_default_133, clone_default_134, clone_default_135
# Graph fragment:
#   %clone_default_134 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_268,), kwargs = {})
#   %clone_default_135 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_270,), kwargs = {})
#   %clone_default_132 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_264,), kwargs = {})
#   %clone_default_133 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_266,), kwargs = {})
triton_poi_fused_57 = async_compile.triton('triton_poi_fused_57', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_57(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/4k/c4kvcwumjsubkrzdzjvyiu253tbegpd35ag2clyvsw3hjbjjyyp6.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_232, triton_kernel_wrapper_mutation_233
# Graph fragment:
#   %triton_kernel_wrapper_mutation_233 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 271, constant_args_idx: 388, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_324, SUM: %as_strided_default_269, SUMSQ: %as_strided_default_271, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_232 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 272, constant_args_idx: 389, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_324, MEAN: %div_39, INVSTD: %rsqrt_13, GAMMA: %primals_82, BETA: %primals_83, Y: %permute_38, X_hat: %permute_39, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_58 = async_compile.triton('triton_poi_fused_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 6576668672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_58(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
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


# kernel path: /tmp/torchinductor_yyu496/e3/ce3u7tdtlnn57alplw23b3g2mr42ni3odepkyv5udp5awxsdnycb.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_232
#   layer2_0_bn2 => full_default_72
#   layer2_0_bn3 => add_56, clamp_min_26, div_39, div_40, mul_92, rsqrt_13, sub_26
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_39 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_269, %full_default_72), kwargs = {})
#   %div_40 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_271, %full_default_72), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_39, %div_39), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_40, %mul_92), kwargs = {})
#   %clamp_min_26 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_26, 0.0), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_26, 1e-05), kwargs = {})
#   %rsqrt_13 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_56,), kwargs = {})
#   %triton_kernel_wrapper_mutation_232 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 272, constant_args_idx: 389, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_324, MEAN: %div_39, INVSTD: %rsqrt_13, GAMMA: %primals_82, BETA: %primals_83, Y: %permute_38, X_hat: %permute_39, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_59 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_59', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_59(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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


# kernel path: /tmp/torchinductor_yyu496/ab/cabhttq6jh5ctairupjcbapqlk6sbcpfmot2ki2c3gwmh4vk3orh.py
# Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer2_0_downsample_0 => convert_element_type_29
# Graph fragment:
#   %convert_element_type_29 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_86, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_60 = async_compile.triton('triton_poi_fused__to_copy_60', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_60', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_60(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hx/chxmk35kamyybzmfamr4etjbuzihd5d4dhlftq2rcyaxajtofehv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_131
# Graph fragment:
#   %clone_default_131 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_262,), kwargs = {})
triton_poi_fused_61 = async_compile.triton('triton_poi_fused_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2466250752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_61(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 822083584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2e/c2ebxkeo6zvsqgoorb2u7vm5oydizlmpzpjyffci52tsp2mo3l5n.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_226
# Graph fragment:
#   %triton_kernel_wrapper_mutation_226 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 395, grid: [(802816, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %add_63, Y_ptr: %permute_42, Mask_prt: %as_strided_default_263, n_elts: 822083584, BLOCK_SIZE: 1024}})
triton_poi_fused_62 = async_compile.triton('triton_poi_fused_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_62(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 822083584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jb/cjboc6t2owceghevd2cnffddu5sxjyejmvxbqlfkcbgt3yc2xhin.py
# Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer2_1_conv1 => avg_pool2d_15, convert_element_type_32
# Graph fragment:
#   %avg_pool2d_15 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_42, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_15, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_63 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 169869312, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_63(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 81
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 9)
    x2 = xindex // 9
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 512)
    y4 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (28 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (29 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (30 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (56 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (57 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (58 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 512*x5 + 41472*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/h4/ch4vmov7ludwylpw24omr6lbtilq2fptsddnrn55uwywoj4435qe.py
# Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer2_1_conv1 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_42, %convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_64 = async_compile.triton('triton_poi_fused_convolution_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3288334336, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_64(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
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


# kernel path: /tmp/torchinductor_yyu496/fg/cfgfk3dq2nt6najptc5phqcer4hyix732wkq52psr3vuemtchtmk.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_123, clone_default_127
# Graph fragment:
#   %clone_default_127 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_254,), kwargs = {})
#   %clone_default_123 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_246,), kwargs = {})
triton_poi_fused_65 = async_compile.triton('triton_poi_fused_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_65', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1027604480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_65(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4i/c4it4kp2srkvgsie5jnaddti3pqfr7m6yv3e4magrxvxozg5l64w.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_122, clone_default_126
# Graph fragment:
#   %clone_default_126 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_252,), kwargs = {})
#   %clone_default_122 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_244,), kwargs = {})
triton_poi_fused_66 = async_compile.triton('triton_poi_fused_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'out_ptr1': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_66', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 128450560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_66(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/6i/c6i2cnrd7ey6pwqwkj3nfdctd7jkvexydq4jfllfj4iiepcje46k.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_120, clone_default_121
# Graph fragment:
#   %clone_default_120 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_240,), kwargs = {})
#   %clone_default_121 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_242,), kwargs = {})
triton_poi_fused_67 = async_compile.triton('triton_poi_fused_67', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_67', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_67(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ge/cgecem7vwbslxuxvmuncifvw3qcmkf3hi5v4mo6kx526gs3i65wh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_101, clone_default_104, clone_default_105
# Graph fragment:
#   %clone_default_104 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_208,), kwargs = {})
#   %clone_default_105 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_210,), kwargs = {})
#   %clone_default_101 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_202,), kwargs = {})
triton_poi_fused_68 = async_compile.triton('triton_poi_fused_68', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_68', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_68(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/qx/cqxf2b7u225vy6xf2epojpizvwcypgbhxnoproe6kr3uwmb5a2dk.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_168, triton_kernel_wrapper_mutation_169
# Graph fragment:
#   %triton_kernel_wrapper_mutation_169 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 315, constant_args_idx: 452, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_595, SUM: %as_strided_default_193, SUMSQ: %as_strided_default_195, M: 1605632, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_168 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 316, constant_args_idx: 453, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_595, MEAN: %div_72, INVSTD: %rsqrt_24, GAMMA: %primals_148, BETA: %primals_149, Y: %permute_70, X_hat: %permute_71, M: 1605632, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_69 = async_compile.triton('triton_poi_fused_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_69', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_69(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
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


# kernel path: /tmp/torchinductor_yyu496/zb/czbf2kk2yt56ufpq33i6fhshduxa265t2ebw4kqyyrr6xuaxdshk.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer3_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_168
#   layer2_0_bn2 => full_default_72
#   layer3_0_bn1 => add_104, clamp_min_48, div_72, div_73, mul_169, rsqrt_24, sub_48
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_72 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_193, %full_default_72), kwargs = {})
#   %div_73 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_195, %full_default_72), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_72, %div_72), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_73, %mul_169), kwargs = {})
#   %clamp_min_48 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_48, 0.0), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_48, 1e-05), kwargs = {})
#   %rsqrt_24 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_104,), kwargs = {})
#   %triton_kernel_wrapper_mutation_168 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 316, constant_args_idx: 453, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_595, MEAN: %div_72, INVSTD: %rsqrt_24, GAMMA: %primals_148, BETA: %primals_149, Y: %permute_70, X_hat: %permute_71, M: 1605632, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_70 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_70', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_70', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_70(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/l5/cl5tqbu74wep5fylrrwbeup2673b4nobpekrw7bvsghawdpjiukg.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_95
# Graph fragment:
#   %clone_default_95 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_190,), kwargs = {})
triton_poi_fused_71 = async_compile.triton('triton_poi_fused_71', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_71', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 154140672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_71(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lx/clxzk7rivyyegpfppwtl2ybrlkmfdiivygq6ofwmkkomxxvpx4gx.py
# Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer3_0_conv2 => convert_element_type_51
# Graph fragment:
#   %convert_element_type_51 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_152, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_72 = async_compile.triton('triton_poi_fused__to_copy_72', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_72', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_72(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/bt/cbtkgdjdrqzevnhqw6nnrd6ga4n4dghyirtwczysax6eprecehof.py
# Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer3_0_conv2 => avg_pool2d_25, convert_element_type_52
# Graph fragment:
#   %avg_pool2d_25 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_72, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_52 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_25, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_73 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_73', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 128}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_73', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 84934656, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_73(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 81
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 9)
    x2 = xindex // 9
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 256)
    y4 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (28 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (29 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (30 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (56 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (57 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (58 + 3*x1 + 84*x2 + 784*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 256*x5 + 20736*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lu/clu4t4dirfcfsegcvgecuyvcpa7ixkskcklq373rpxhrk62tifq6.py
# Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_conv2 => convolution_25
# Graph fragment:
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_72, %convert_element_type_51, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_74 = async_compile.triton('triton_poi_fused_convolution_74', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_74', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_74(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
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


# kernel path: /tmp/torchinductor_yyu496/yt/cytijs4bdghlufhbmytrjt3mll75567jyogezsoeslpengcrnfcn.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_162, triton_kernel_wrapper_mutation_163
# Graph fragment:
#   %triton_kernel_wrapper_mutation_163 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 319, constant_args_idx: 458, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_620, SUM: %as_strided_default_187, SUMSQ: %as_strided_default_189, M: 401408, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_162 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 320, constant_args_idx: 459, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_620, MEAN: %div_75, INVSTD: %rsqrt_25, GAMMA: %primals_154, BETA: %primals_155, Y: %permute_73, X_hat: %permute_74, M: 401408, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_75 = async_compile.triton('triton_poi_fused_75', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_75', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_75(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
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


# kernel path: /tmp/torchinductor_yyu496/bm/cbmzznnnz54njdtjzb2potlws63hohpt6rchobvf375wrxlelowl.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_162
#   layer3_0_bn2 => add_108, clamp_min_50, div_75, div_76, full_default_148, mul_176, rsqrt_25, sub_50
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_75 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_187, %full_default_148), kwargs = {})
#   %div_76 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_189, %full_default_148), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_75, %div_75), kwargs = {})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_76, %mul_176), kwargs = {})
#   %clamp_min_50 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_50, 0.0), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_50, 1e-05), kwargs = {})
#   %rsqrt_25 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_108,), kwargs = {})
#   %triton_kernel_wrapper_mutation_162 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 320, constant_args_idx: 459, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_620, MEAN: %div_75, INVSTD: %rsqrt_25, GAMMA: %primals_154, BETA: %primals_155, Y: %permute_73, X_hat: %permute_74, M: 401408, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/c5/cc5cyzkljd6gl7jcqimejiybjpv5caw4np4gshflkztp6wythnxd.py
# Topologically Sorted Source Nodes: [layer3_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer3_0_relu_1 => full_default_150
# Graph fragment:
#   %full_default_150 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([2048, 256, 14, 14], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_77 = async_compile.triton('triton_poi_fused_zeros_77', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_77', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_77(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mo/cmouluf6zgdevhw76y24pjjadrvj3ftu2loapd22mg4nx5d2pcjx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_92
# Graph fragment:
#   %clone_default_92 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_184,), kwargs = {})
triton_poi_fused_78 = async_compile.triton('triton_poi_fused_78', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_78', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 308281344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_78(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/dq/cdqpxdounc6oabc7qfp7lsdnwdalrysj3k75ofzlk3ao5ssxrr7j.py
# Topologically Sorted Source Nodes: [layer3_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer3_0_relu_1 => full_default_151
# Graph fragment:
#   %full_default_151 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([200704, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_79 = async_compile.triton('triton_poi_fused_zeros_79', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_79', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25690112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_79(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wv/cwvj2hzds652363asgtgluh5xrhztlyaywyeoyfbd4r3xxoqnjy5.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_91
# Graph fragment:
#   %clone_default_91 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_182,), kwargs = {})
triton_poi_fused_80 = async_compile.triton('triton_poi_fused_80', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_80', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 38535168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_80(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xn/cxngyro3c4bifsbxcrlf3rx7jwkn4z7juypyb3basoczjdosaxdf.py
# Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer3_0_conv3 => convert_element_type_53
# Graph fragment:
#   %convert_element_type_53 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_158, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_81 = async_compile.triton('triton_poi_fused__to_copy_81', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_81', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_81(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/su/csumss7xjl5kwlhzayyq47lbipocbexshisytfaxdcrochtwefkg.py
# Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer3_0_conv3 => avg_pool2d_26, convert_element_type_54
# Graph fragment:
#   %avg_pool2d_26 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_75, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_54 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_26, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_82 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_82', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_82', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 16777216, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_82(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 4)
    x2 = xindex // 4
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 256)
    y4 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (14 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (15 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (16 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (28 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (29 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (30 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 256*x5 + 4096*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/52/c52eqy5il2jmpghzrwvt3pxfbn3qmxxxqgrya4tb5qlsuxgnsxhl.py
# Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_conv3 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_75, %convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_83 = async_compile.triton('triton_poi_fused_convolution_83', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_83', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_83(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 50176*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/p4/cp4qwbt64xzk3ipto7pp7w3c2s4d4clpbhubgfspwwwsu4ydonyh.py
# Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer3_0_bn3 => full_default_152
# Graph fragment:
#   %full_default_152 : [num_users=15] = call_function[target=torch.ops.aten.full.default](args = ([1024], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_84 = async_compile.triton('triton_poi_fused_zeros_84', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_84', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_84(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ui/cuiod5nmfb4d67nnrync5y5ppbybpjnvah3od33mbeefgqfihorc.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_87, clone_default_88, clone_default_89, clone_default_90
# Graph fragment:
#   %clone_default_89 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_178,), kwargs = {})
#   %clone_default_90 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_180,), kwargs = {})
#   %clone_default_87 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_174,), kwargs = {})
#   %clone_default_88 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_176,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_85', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_85(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/2q/c2q5ek5lp6lcykdpxbbnxj64qcghs6r3msyoeue6damcd2ate72x.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_156, triton_kernel_wrapper_mutation_157
# Graph fragment:
#   %triton_kernel_wrapper_mutation_157 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 323, constant_args_idx: 464, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_645, SUM: %as_strided_default_179, SUMSQ: %as_strided_default_181, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_156 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 324, constant_args_idx: 465, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_645, MEAN: %div_78, INVSTD: %rsqrt_26, GAMMA: %primals_160, BETA: %primals_161, Y: %permute_76, X_hat: %permute_77, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_86 = async_compile.triton('triton_poi_fused_86', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_86', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_86(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
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


# kernel path: /tmp/torchinductor_yyu496/7g/c7gmns6lsjaqllkbe4clcyht3dnhkcxzjzdv3n226gaykovq4ocy.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_156
#   layer3_0_bn2 => full_default_148
#   layer3_0_bn3 => add_112, clamp_min_52, div_78, div_79, mul_183, rsqrt_26, sub_52
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_78 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_179, %full_default_148), kwargs = {})
#   %div_79 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_181, %full_default_148), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_78, %div_78), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_79, %mul_183), kwargs = {})
#   %clamp_min_52 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_52, 0.0), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_52, 1e-05), kwargs = {})
#   %rsqrt_26 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_112,), kwargs = {})
#   %triton_kernel_wrapper_mutation_156 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 324, constant_args_idx: 465, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_645, MEAN: %div_78, INVSTD: %rsqrt_26, GAMMA: %primals_160, BETA: %primals_161, Y: %permute_76, X_hat: %permute_77, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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


# kernel path: /tmp/torchinductor_yyu496/2q/c2q7p3rgntnwxr25kr65btwxdczbd75gagfvpe46pfhux2r34wlz.py
# Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer3_0_downsample_0 => convert_element_type_55
# Graph fragment:
#   %convert_element_type_55 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_164, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_88 = async_compile.triton('triton_poi_fused__to_copy_88', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_88', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_88(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/45/c455cfadwhpftktpoxqsqr47hfc5xozwas7rxnys3pfcrfthvwtk.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_150
# Graph fragment:
#   %triton_kernel_wrapper_mutation_150 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 471, grid: [(401408, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %add_119, Y_ptr: %permute_80, Mask_prt: %as_strided_default_173, n_elts: 411041792, BLOCK_SIZE: 1024}})
triton_poi_fused_89 = async_compile.triton('triton_poi_fused_89', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_89', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_89(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/nz/cnzcrkafhugnlmxrpyadoyc2ptep3pe7az3vdvi67mnlkhbilsn6.py
# Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer3_1_conv1 => avg_pool2d_28, convert_element_type_58
# Graph fragment:
#   %avg_pool2d_28 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_80, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_58 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_28, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_90 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_90', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_90', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 67108864, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_90(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 4)
    x2 = xindex // 4
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 1024)
    y4 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (14 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (15 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (16 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (28 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (29 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (30 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 1024*x5 + 16384*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/zx/czx5nem6wu57ci3wyr6tbzb74jyoo73fqnwftp5wrj66ow2vkgf7.py
# Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer3_1_conv1 => convolution_28
# Graph fragment:
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_80, %convert_element_type_57, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_91 = async_compile.triton('triton_poi_fused_convolution_91', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_91', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_91(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
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


# kernel path: /tmp/torchinductor_yyu496/jh/cjhyjqalgzewhl4ftshzjogoh7swbldcro7llg7lqwo5gk6k7234.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_78, clone_default_82
# Graph fragment:
#   %clone_default_82 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_164,), kwargs = {})
#   %clone_default_78 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_156,), kwargs = {})
triton_poi_fused_92 = async_compile.triton('triton_poi_fused_92', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_92', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 513802240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_92(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/q2/cq2gkexstlqvs3ldqfhza3nmlywklndstsf6wmafzb2f2xkq2ehq.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_77, clone_default_81
# Graph fragment:
#   %clone_default_81 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_162,), kwargs = {})
#   %clone_default_77 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_154,), kwargs = {})
triton_poi_fused_93 = async_compile.triton('triton_poi_fused_93', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_93', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64225280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_93(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hr/chrolzjnqopugpe6et7nn3ysbj46f375gqscbgimfk5sn6rwrwlo.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_75, clone_default_76
# Graph fragment:
#   %clone_default_75 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_150,), kwargs = {})
#   %clone_default_76 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_152,), kwargs = {})
triton_poi_fused_94 = async_compile.triton('triton_poi_fused_94', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_94', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_94(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wq/cwqrvr7k7sswmi3ovdfe4tcoyqtq6pvrwmoutpqn7gbodjk3me2p.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_32, clone_default_35, clone_default_36
# Graph fragment:
#   %clone_default_35 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_70,), kwargs = {})
#   %clone_default_36 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_72,), kwargs = {})
#   %clone_default_32 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_64,), kwargs = {})
triton_poi_fused_95 = async_compile.triton('triton_poi_fused_95', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_95', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_95(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/yu/cyu6dgbx3khit676oxka73y3ndwng7cfm7dnrzappcrtvkiobwap.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_31
# Graph fragment:
#   %clone_default_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_62,), kwargs = {})
triton_poi_fused_96 = async_compile.triton('triton_poi_fused_96', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_96', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_96(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/n7/cn7ljxlviev423zvaqjtjues2ao2s3t2qo54owiyorpg7gyvqenm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_56, triton_kernel_wrapper_mutation_57
# Graph fragment:
#   %triton_kernel_wrapper_mutation_57 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 391, constant_args_idx: 564, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1066, SUM: %as_strided_default_59, SUMSQ: %as_strided_default_61, M: 401408, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_56 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 392, constant_args_idx: 565, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1066, MEAN: %div_129, INVSTD: %rsqrt_43, GAMMA: %primals_262, BETA: %primals_263, Y: %permute_126, X_hat: %permute_127, M: 401408, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_97 = async_compile.triton('triton_poi_fused_97', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_97', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_97(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
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


# kernel path: /tmp/torchinductor_yyu496/wl/cwl32pf5rzbff6vpyxk7dqewvhxwcna43o4mztyl6tpupplnb4k5.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer4_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_56
#   layer3_0_bn2 => full_default_148
#   layer4_0_bn1 => add_186, clamp_min_86, div_129, div_130, mul_302, rsqrt_43, sub_86
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_129 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_59, %full_default_148), kwargs = {})
#   %div_130 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_61, %full_default_148), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_129, %div_129), kwargs = {})
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_130, %mul_302), kwargs = {})
#   %clamp_min_86 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_86, 0.0), kwargs = {})
#   %add_186 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_86, 1e-05), kwargs = {})
#   %rsqrt_43 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_186,), kwargs = {})
#   %triton_kernel_wrapper_mutation_56 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 392, constant_args_idx: 565, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1066, MEAN: %div_129, INVSTD: %rsqrt_43, GAMMA: %primals_262, BETA: %primals_263, Y: %permute_126, X_hat: %permute_127, M: 401408, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_98 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_98', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_98', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_98(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/7p/c7prjabyxdgb5p6o22mb55hzsvafakhab3ckdl3vemqf6y5erhdo.py
# Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer4_0_conv2 => convert_element_type_89
# Graph fragment:
#   %convert_element_type_89 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_266, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_99 = async_compile.triton('triton_poi_fused__to_copy_99', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_99', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9437184, 'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_99(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/74/c74twr3ogqttlefjqnphnng5kl2nl2c65yysjqskl5ydq2cgcuoe.py
# Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer4_0_conv2 => avg_pool2d_44, convert_element_type_90
# Graph fragment:
#   %avg_pool2d_44 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_128, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_90 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_44, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_100 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_100', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_100', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 33554432, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_100(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 4)
    x2 = xindex // 4
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 512)
    y4 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (14 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (15 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (16 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (28 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (29 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (30 + 3*x1 + 42*x2 + 196*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 512*x5 + 8192*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/u7/cu7hee4sfpfujk6rtemnz6z75k2aas3pb2sg7lhvqjsjygvhwdgr.py
# Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_conv2 => convolution_44
# Graph fragment:
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_128, %convert_element_type_89, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_101 = async_compile.triton('triton_poi_fused_convolution_101', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_101', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_101(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
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


# kernel path: /tmp/torchinductor_yyu496/ns/cnsxtmdingbowdwuuvxy7cx4c765wjajuoezo22kpsvdd6y3azot.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_50, triton_kernel_wrapper_mutation_51
# Graph fragment:
#   %triton_kernel_wrapper_mutation_51 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 395, constant_args_idx: 570, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1091, SUM: %as_strided_default_53, SUMSQ: %as_strided_default_55, M: 100352, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_50 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 396, constant_args_idx: 571, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1091, MEAN: %div_132, INVSTD: %rsqrt_44, GAMMA: %primals_268, BETA: %primals_269, Y: %permute_129, X_hat: %permute_130, M: 100352, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
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


# kernel path: /tmp/torchinductor_yyu496/5e/c5ekyfc4dj6vwu4jpg2e74mhki7tjapeh2iprbha2h7tzyfzuldc.py
# Topologically Sorted Source Nodes: [layer4_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_50
#   layer4_0_bn2 => add_190, clamp_min_88, div_132, div_133, full_default_260, mul_309, rsqrt_44, sub_88
# Graph fragment:
#   %full_default_260 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_132 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_53, %full_default_260), kwargs = {})
#   %div_133 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_55, %full_default_260), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_132, %div_132), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_133, %mul_309), kwargs = {})
#   %clamp_min_88 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_88, 0.0), kwargs = {})
#   %add_190 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_88, 1e-05), kwargs = {})
#   %rsqrt_44 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_190,), kwargs = {})
#   %triton_kernel_wrapper_mutation_50 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 396, constant_args_idx: 571, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1091, MEAN: %div_132, INVSTD: %rsqrt_44, GAMMA: %primals_268, BETA: %primals_269, Y: %permute_129, X_hat: %permute_130, M: 100352, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/5n/c5nhlipwdl6bedcrrqbxt4ztqkzsdelev5f4vcrimzalofsguhje.py
# Topologically Sorted Source Nodes: [layer4_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer4_0_relu_1 => full_default_262
# Graph fragment:
#   %full_default_262 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([2048, 512, 7, 7], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_104 = async_compile.triton('triton_poi_fused_zeros_104', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_104', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_104(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/x7/cx7dthclqruyesxwbeqeq4dud4kpcjnb5h4runairxfjomz4d3ze.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_25
# Graph fragment:
#   %clone_default_25 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_50,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_105', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 154140672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_105(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/zj/czjznf2brxbty24lazo3blmz5titygqujhznjnnbbeoi6dcacz7g.py
# Topologically Sorted Source Nodes: [layer4_0_relu_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer4_0_relu_1 => full_default_263
# Graph fragment:
#   %full_default_263 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([100352, 16], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_106 = async_compile.triton('triton_poi_fused_zeros_106', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_106', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12845056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_106(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/tr/ctrr4svhqroteklro66abrbvxvwjb6q4wpqxvoeszdkosbe7kgjh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_24
# Graph fragment:
#   %clone_default_24 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_48,), kwargs = {})
triton_poi_fused_107 = async_compile.triton('triton_poi_fused_107', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_107', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 19267584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_107(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vg/cvgrdyx2ulave6g4cffun6pd3x54dwymeaspjjnxvdinvx6sagnf.py
# Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer4_0_conv3 => convert_element_type_91
# Graph fragment:
#   %convert_element_type_91 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_272, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_108 = async_compile.triton('triton_poi_fused__to_copy_108', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_108', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_108(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/s4/cs4j2iu2q62mfkgludtzm5sy7jecldv5dr3bolw5xicy3qskmk7l.py
# Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer4_0_conv3 => avg_pool2d_45, convert_element_type_92
# Graph fragment:
#   %avg_pool2d_45 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_131, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_92 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_45, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_109 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_109', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_109', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 8388608, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_109(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 2)
    x2 = xindex // 2
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 512)
    y4 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (7 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (8 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (9 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (14 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (15 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (16 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 512*x5 + 2048*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wd/cwdnafxzqqzzrhhcrcqkaxsab5yymqphlnzbcpx6fqmtfspuc6rh.py
# Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_conv3 => convolution_45
# Graph fragment:
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_131, %convert_element_type_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_110 = async_compile.triton('triton_poi_fused_convolution_110', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_110', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_110(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 25088*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5i/c5i4lkogeu74wjtwrrhm4gzgrt2unojjez74j7brizgpjuooozpw.py
# Topologically Sorted Source Nodes: [layer4_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer4_0_bn3 => full_default_264
# Graph fragment:
#   %full_default_264 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([2048], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_111 = async_compile.triton('triton_poi_fused_zeros_111', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_111', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_111(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qa/cqa3tdkhvxuug3i5nc2r5tw7fd6hmhh3qju72c5a5vlnlqvv6wqy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_20, clone_default_21, clone_default_22, clone_default_23
# Graph fragment:
#   %clone_default_22 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_44,), kwargs = {})
#   %clone_default_23 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_46,), kwargs = {})
#   %clone_default_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_40,), kwargs = {})
#   %clone_default_21 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_42,), kwargs = {})
triton_poi_fused_112 = async_compile.triton('triton_poi_fused_112', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_112', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 73728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_112(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/5b/c5buuiu5uw6e6sbjdqehs3nbk6fsbosdenbrhvupgihtc22435df.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_44, triton_kernel_wrapper_mutation_45
# Graph fragment:
#   %triton_kernel_wrapper_mutation_45 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 399, constant_args_idx: 576, grid: [(2048, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1116, SUM: %as_strided_default_45, SUMSQ: %as_strided_default_47, M: 100352, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_44 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 400, constant_args_idx: 577, grid: [(2048, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1116, MEAN: %div_135, INVSTD: %rsqrt_45, GAMMA: %primals_274, BETA: %primals_275, Y: %permute_132, X_hat: %permute_133, M: 100352, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_113 = async_compile.triton('triton_poi_fused_113', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4194304, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_113', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_113(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4194304
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


# kernel path: /tmp/torchinductor_yyu496/65/c65lflcyckgyh7nwi7rm3nrfhyt5gjrt6ewmvqvs7eekt4r35tro.py
# Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_44
#   layer4_0_bn2 => full_default_260
#   layer4_0_bn3 => add_194, clamp_min_90, div_135, div_136, mul_316, rsqrt_45, sub_90
# Graph fragment:
#   %full_default_260 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_135 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_45, %full_default_260), kwargs = {})
#   %div_136 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_47, %full_default_260), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_135, %div_135), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_136, %mul_316), kwargs = {})
#   %clamp_min_90 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_90, 0.0), kwargs = {})
#   %add_194 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_min_90, 1e-05), kwargs = {})
#   %rsqrt_45 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_194,), kwargs = {})
#   %triton_kernel_wrapper_mutation_44 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 400, constant_args_idx: 577, grid: [(2048, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X: %view_1116, MEAN: %div_135, INVSTD: %rsqrt_45, GAMMA: %primals_274, BETA: %primals_275, Y: %permute_132, X_hat: %permute_133, M: 100352, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_add_clamp_div_mul_rsqrt_sub_114 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_rsqrt_sub_114', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_rsqrt_sub_114', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsqrt_sub_114(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
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


# kernel path: /tmp/torchinductor_yyu496/mw/cmwsqgs4f244x2pzeohh36simczpddttef6o6eru7yuubttda3hn.py
# Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   layer4_0_downsample_0 => convert_element_type_93
# Graph fragment:
#   %convert_element_type_93 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_278, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_115 = async_compile.triton('triton_poi_fused__to_copy_115', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_115', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_115(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fi/cfi6jmwq3jocmab4b5wptwzo7pnczdpo7x7hcu6nm2go4umso5bb.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_38
# Graph fragment:
#   %triton_kernel_wrapper_mutation_38 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 7, constant_args_idx: 583, grid: [(200704, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X_ptr: %add_201, Y_ptr: %permute_136, Mask_prt: %as_strided_default_39, n_elts: 205520896, BLOCK_SIZE: 1024}})
triton_poi_fused_116 = async_compile.triton('triton_poi_fused_116', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_116', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_116(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/sc/csc4hqgukgvasraovyox2zxowyn6obmrwrn2cx5s5dygxn6nfj2i.py
# Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
# Source node to ATen node mapping:
#   layer4_1_conv1 => avg_pool2d_47, convert_element_type_96
# Graph fragment:
#   %avg_pool2d_47 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_136, [3, 3], [3, 3]), kwargs = {})
#   %convert_element_type_96 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%avg_pool2d_47, torch.float8_e4m3fn), kwargs = {})
triton_poi_fused__to_copy_avg_pool2d_117 = async_compile.triton('triton_poi_fused__to_copy_avg_pool2d_117', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4194304, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*fp8e4nv', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_avg_pool2d_117', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 33554432, 'x': 0}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy_avg_pool2d_117(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4194304
    xnumel = 4
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 2)
    x2 = xindex // 2
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 2048)
    y4 = yindex // 2048
    tmp0 = tl.load(in_ptr0 + (3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (1 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (2 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (7 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (8 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (9 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (14 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (15 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (16 + 3*x1 + 21*x2 + 49*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float8e4nv)
    tl.store(out_ptr1 + (y3 + 2048*x5 + 8192*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4s/c4sxodjiwj2w3fzopvue7f6k2hnkhu3t5zgdwu3buakec6b2shyf.py
# Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   layer4_1_conv1 => convolution_47
# Graph fragment:
#   %convolution_47 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_136, %convert_element_type_95, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_118 = async_compile.triton('triton_poi_fused_convolution_118', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4194304, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_118', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 0}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_118(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4194304
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


# kernel path: /tmp/torchinductor_yyu496/4i/c4ib5wrrhlusy234n7gommbafq2ejnxqx7gfuotfskucwbjikhaa.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_11, clone_default_15
# Graph fragment:
#   %clone_default_15 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_30,), kwargs = {})
#   %clone_default_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_22,), kwargs = {})
triton_poi_fused_119 = async_compile.triton('triton_poi_fused_119', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_119', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256901120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_119(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vn/cvnhdmpembiczkw6yqryfuenxn2zm3zdb4kkutmyfypcl273qqzm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_10, clone_default_14
# Graph fragment:
#   %clone_default_14 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_28,), kwargs = {})
#   %clone_default_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_20,), kwargs = {})
triton_poi_fused_120 = async_compile.triton('triton_poi_fused_120', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_120', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32112640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_120(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vk/cvkji253nqfoiadqbhsvcmsix4jgzgiugmnuttn4rumh5kx5xpbn.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_8, clone_default_9
# Graph fragment:
#   %clone_default_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_16,), kwargs = {})
#   %clone_default_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_18,), kwargs = {})
triton_poi_fused_121 = async_compile.triton('triton_poi_fused_121', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_121', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 40960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_121(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wu/cwuzf5pauuthmwo3544o66j77i42ur74jiff4e4vygp7oav7qywd.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_1, clone_default_4, clone_default_5
# Graph fragment:
#   %clone_default_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_8,), kwargs = {})
#   %clone_default_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_10,), kwargs = {})
#   %clone_default_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_2,), kwargs = {})
triton_poi_fused_122 = async_compile.triton('triton_poi_fused_122', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_122', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_122(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/vg/cvgm2avzupjnmjnrfzonn2i4vvh3omamo7oeyax6j5y26td5dplk.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default
# Graph fragment:
#   %clone_default : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default,), kwargs = {})
triton_poi_fused_123 = async_compile.triton('triton_poi_fused_123', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_123', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_123(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gv/cgvrbkealbuhyqtrkvn5gyphrbd7mlzv23oofsyrsrg5w2o672dx.py
# Topologically Sorted Source Nodes: [avgpool, flatten], Original ATen: [aten.mean, aten.view]
# Source node to ATen node mapping:
#   avgpool => mean
#   flatten => view_1303
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%permute_154, [-1, -2], True), kwargs = {})
#   %view_1303 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mean, [2048, 2048]), kwargs = {})
triton_per_fused_mean_view_124 = async_compile.triton('triton_per_fused_mean_view_124', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4194304, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_view_124', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16777216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_mean_view_124(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4194304
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


# kernel path: /tmp/torchinductor_yyu496/tt/cttiw2mo75thagbkf6ulgur7hi2h736t5sfj5s5tfia6xt64ebce.py
# Topologically Sorted Source Nodes: [fc], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   fc => convert_element_type_107
# Graph fragment:
#   %convert_element_type_107 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_320, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_125 = async_compile.triton('triton_poi_fused__to_copy_125', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_125', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1638400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_125(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hb/chbgrc5hxdlgcufpaofdzkjpef6teoq7ujcptzhzutuc5vgrwpaf.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, 1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_3, %add), kwargs = {})
triton_poi_fused_add_copy__126 = async_compile.triton('triton_poi_fused_add_copy__126', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__126', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__126(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/2d/c2doummvwmanup64puadubjsmnq7cpvvp6oookbpsqpsaodwuhqp.py
# Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   bn1 => add_2, add_3, clamp_min, div, div_1, full_default_2, mul_1, mul_3, mul_4, mul_5, mul_6, sub
# Graph fragment:
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 25690112.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_357, %full_default_2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_359, %full_default_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %div), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %mul_1), kwargs = {})
#   %clamp_min : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, 0.9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 0.1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_7, 0.9), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min, 0.1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_6, %add_2), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_7, %add_3), kwargs = {})
triton_poi_fused_add_copy__div_mul_sub_127 = async_compile.triton('triton_poi_fused_add_copy__div_mul_sub_127', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__div_mul_sub_127', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__div_mul_sub_127(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 3.8925482302295915e-08
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
    tmp17 = tmp16 * tmp6
    tmp18 = tmp10 + tmp17
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr3 + (x0), tmp18, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ps/cpsyoshxnhimeerjvbyevgzdql5k5rsnroan7xrsslsivcugt4vx.py
# Topologically Sorted Source Nodes: [layer1_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer1_0_bn1 => add_6, add_7, clamp_min_2, div_3, div_4, full_default_8, full_default_9, mul_10, mul_11, mul_12, mul_13, mul_8, mul_9, sub_2
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 6422528.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_3 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_351, %full_default_8), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_353, %full_default_8), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %div_3), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_4, %mul_8), kwargs = {})
#   %clamp_min_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %full_default_9 : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000001192092896), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_2, %full_default_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, 0.9), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, 0.1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_13, 0.9), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 0.1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %copy__4 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_12, %add_6), kwargs = {})
#   %copy__5 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_13, %add_7), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_128 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_128', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_128', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_128(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/52/c522padcdagnjqi3zjok7ymaridlpm6hlmcwr2lww4t4vhxshzph.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer1_0_bn1 => full_default_8, full_default_9
#   layer1_0_bn3 => add_14, add_15, clamp_min_6, div_10, div_9, mul_22, mul_23, mul_24, mul_25, mul_26, mul_27, sub_6
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 6422528.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_9 : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000001192092896), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
triton_poi_fused_add_clamp_copy__div_mul_sub_129 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_129', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_129', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_129(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/3a/c3almta3yvb5pnpir345lao37kkgzmkmt7dkmvygi4ecdds2f6mc.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer2_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer1_0_bn1 => full_default_8, full_default_9
#   layer2_0_bn1 => add_49, add_50, clamp_min_22, div_33, div_34, mul_78, mul_79, mul_80, mul_81, mul_82, mul_83, sub_22
# Graph fragment:
#   %full_default_8 : [num_users=22] = call_function[target=torch.ops.aten.full.default](args = ([], 6422528.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_9 : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000001192092896), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
triton_poi_fused_add_clamp_copy__div_mul_sub_130 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_130', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_130', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_130(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/cp/ccpefewuymfmufnuooixrvnae7kmrckjfvo55d5zzg5wci424sxf.py
# Topologically Sorted Source Nodes: [layer2_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer2_0_bn2 => add_53, add_54, clamp_min_24, div_36, div_37, full_default_72, full_default_73, mul_85, mul_86, mul_87, mul_88, mul_89, mul_90, sub_24
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_36 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_277, %full_default_72), kwargs = {})
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_279, %full_default_72), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_36, %div_36), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_37, %mul_85), kwargs = {})
#   %clamp_min_24 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_24, 0.0), kwargs = {})
#   %full_default_73 : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000005960464478), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_24, %full_default_73), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_78, 0.9), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_36, 0.1), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %mul_88), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_79, 0.9), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, 0.1), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %mul_90), kwargs = {})
#   %copy__37 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_78, %add_53), kwargs = {})
#   %copy__38 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_79, %add_54), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_131 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_131', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_131', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_131(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/sr/csrpdrrjijeml2wuocm7ydnmptjftscz6ua6dvkwo2p3tkna4pdx.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer2_0_bn2 => full_default_72, full_default_73
#   layer2_0_bn3 => add_57, add_58, clamp_min_26, div_39, div_40, mul_92, mul_93, mul_94, mul_95, mul_96, mul_97, sub_26
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_73 : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000005960464478), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
triton_poi_fused_add_clamp_copy__div_mul_sub_132 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_132', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_132', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_132(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/zq/czqer5yv7tsiq7qs52dv4dxe7pot6toujf7wyvkxsqfg3nn3t2zf.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer3_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer2_0_bn2 => full_default_72, full_default_73
#   layer3_0_bn1 => add_105, add_106, clamp_min_48, div_72, div_73, mul_169, mul_170, mul_171, mul_172, mul_173, mul_174, sub_48
# Graph fragment:
#   %full_default_72 : [num_users=26] = call_function[target=torch.ops.aten.full.default](args = ([], 1605632.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_73 : [num_users=13] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000005960464478), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
triton_poi_fused_add_clamp_copy__div_mul_sub_133 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_133', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_133', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_133(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/nf/cnf4lz7e2n5nv5cfvqdbxqjmnujbpdqra7mtyf2kynkogpmpthco.py
# Topologically Sorted Source Nodes: [layer3_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer3_0_bn2 => add_109, add_110, clamp_min_50, div_75, div_76, full_default_148, full_default_149, mul_176, mul_177, mul_178, mul_179, mul_180, mul_181, sub_50
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_75 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_187, %full_default_148), kwargs = {})
#   %div_76 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_189, %full_default_148), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_75, %div_75), kwargs = {})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_76, %mul_176), kwargs = {})
#   %clamp_min_50 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_50, 0.0), kwargs = {})
#   %full_default_149 : [num_users=19] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000025033950806), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_177 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_50, %full_default_149), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_156, 0.9), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_75, 0.1), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_178, %mul_179), kwargs = {})
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_157, 0.9), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_177, 0.1), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_180, %mul_181), kwargs = {})
#   %copy__76 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_156, %add_109), kwargs = {})
#   %copy__77 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_157, %add_110), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_134 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_134', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_134', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_134(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/5e/c5eyqslcylxkg3skar7xkjzncezfptk5kgpd7znkyr6fevcc3icy.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer3_0_bn2 => full_default_148, full_default_149
#   layer3_0_bn3 => add_113, add_114, clamp_min_52, div_78, div_79, mul_183, mul_184, mul_185, mul_186, mul_187, mul_188, sub_52
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_149 : [num_users=19] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000025033950806), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
triton_poi_fused_add_clamp_copy__div_mul_sub_135 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_135', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_135', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_135(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/u6/cu6sgc3hsgtgotkuitdsz2zyg6a72ypk7zee56r3jjudpbwvrack.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer4_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer3_0_bn2 => full_default_148, full_default_149
#   layer4_0_bn1 => add_187, add_188, clamp_min_86, div_129, div_130, mul_302, mul_303, mul_304, mul_305, mul_306, mul_307, sub_86
# Graph fragment:
#   %full_default_148 : [num_users=38] = call_function[target=torch.ops.aten.full.default](args = ([], 401408.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_149 : [num_users=19] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000025033950806), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
triton_poi_fused_add_clamp_copy__div_mul_sub_136 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_136', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_136', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_136(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/do/cdomlylsm2h6sbdbbsllyelveg2l5ebzrnsm6a5vpgpnuojzsext.py
# Topologically Sorted Source Nodes: [layer4_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer4_0_bn2 => add_191, add_192, clamp_min_88, div_132, div_133, full_default_260, full_default_261, mul_309, mul_310, mul_311, mul_312, mul_313, mul_314, sub_88
# Graph fragment:
#   %full_default_260 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %div_132 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_53, %full_default_260), kwargs = {})
#   %div_133 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%as_strided_default_55, %full_default_260), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_132, %div_132), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_133, %mul_309), kwargs = {})
#   %clamp_min_88 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_88, 0.0), kwargs = {})
#   %full_default_261 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000100135803223), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_min_88, %full_default_261), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_270, 0.9), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_132, 0.1), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %mul_312), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_271, 0.9), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, 0.1), kwargs = {})
#   %add_192 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_313, %mul_314), kwargs = {})
#   %copy__133 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_270, %add_191), kwargs = {})
#   %copy__134 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_271, %add_192), kwargs = {})
triton_poi_fused_add_clamp_copy__div_mul_sub_137 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_137', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_137', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_137(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/i7/ci7kfppswhxdwhrcarro74xhvxnq6ikls6oqsativl2zezrzvefd.py
# Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   layer4_0_bn2 => full_default_260, full_default_261
#   layer4_0_bn3 => add_195, add_196, clamp_min_90, div_135, div_136, mul_316, mul_317, mul_318, mul_319, mul_320, mul_321, sub_90
# Graph fragment:
#   %full_default_260 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 100352.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_261 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0000100135803223), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
triton_poi_fused_add_clamp_copy__div_mul_sub_138 = async_compile.triton('triton_poi_fused_add_clamp_copy__div_mul_sub_138', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_copy__div_mul_sub_138', 'mutated_arg_names': ['in_ptr0', 'in_ptr2', 'out_ptr1', 'out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_copy__div_mul_sub_138(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (2048, 3, 224, 224), (150528, 50176, 224, 1))
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
        buf1 = empty_strided_cuda((2048, 3, 74, 74), (16428, 1, 222, 3), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy, aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_0.run(primals_2, buf1, 6144, 5476, stream=stream0)
        buf2 = empty_strided_cuda((602112, 32), (32, 1), torch.int32)
        buf3 = empty_strided_cuda((602112, ), (1, ), torch.bfloat16)
        buf4 = empty_strided_cuda((602112, ), (1, ), torch.bfloat16)
        buf5 = empty_strided_cuda((602112, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf5, 308281344, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(buf5, buf2, buf3, buf4, 512, 1, 32, 1, 2, 16, 32, 3, 602112, 1, 1, stream=stream0)
        buf9 = reinterpret_tensor(buf5, (2048, 3, 224, 224), (150528, 1, 672, 3), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(primals_2, buf9, 6144, 50176, stream=stream0)
        del primals_2
        buf10 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.bfloat16)
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_1, buf10, 192, 49, stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf11 = extern_kernels.convolution(buf9, buf10, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (2048, 64, 112, 112), (802816, 1, 7168, 64), 'torch.ops.aten.convolution.default')
        del buf10
        del buf9
        buf12 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_4.run(buf12, 64, stream=stream0)
        buf13 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf14 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf53 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf54 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf90 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf91 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(buf12, buf13, buf14, buf53, buf54, buf90, buf91, 64, stream=stream0)
        buf15 = empty_strided_cuda((2048, 64, 12544), (802816, 12544, 1), torch.bfloat16)
        buf21 = empty_strided_cuda((2048, 64, 12544), (802816, 12544, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(buf11, buf15, buf21, 131072, 12544, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_1.run(buf15, buf13, buf14, 25690112, 12544, 802816, 12544, 1024, 64, 25088, 1, stream=stream0)
        buf18 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf22 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_rsqrt_sub_7.run(buf14, buf13, buf18, buf22, 64, stream=stream0)
        buf19 = buf15; del buf15  # reuse
        buf20 = reinterpret_tensor(buf11, (2048, 64, 12544), (802816, 12544, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_2.run(buf21, buf22, buf18, primals_4, primals_5, buf19, buf20, 25690112, 12544, 802816, 12544, 1024, 64, 25088, 1, stream=stream0)
        del primals_5
        buf25 = empty_strided_cuda((3211264, 32), (32, 1), torch.int32)
        buf26 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        buf27 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf20, (3211264, 512), (512, 1), 0), buf25, buf26, buf27, 512, 1, 32, 1, 2, 16, 32, 3, 3211264, 1, 1, stream=stream0)
        buf31 = reinterpret_tensor(buf20, (2048, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf20  # reuse
        buf33 = reinterpret_tensor(buf21, (2048, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [relu, ], Original ATen: [aten.empty_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_empty_like_8.run(buf31, buf33, 1644167168, stream=stream0)
        buf34 = empty_strided_cuda((2048, 64, 112, 112), (802816, 12544, 112, 1), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(buf34, 1644167168, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf19, (2048, 64, 112, 112), (802816, 12544, 112, 1), 0), buf33, buf34, 1644167168, 1024, 1605632, 1, 1, stream=stream0)
        buf37 = empty_strided_cuda((3211264, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_10.run(buf37, 51380224, stream=stream0)
        buf38 = empty_strided_cuda((51380224, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf37, buf38, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf34, (3211264, 512), (512, 1), 0), reinterpret_tensor(buf38, (3211264, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 3211264, 1, 1, stream=stream0)
        buf40 = empty_strided_cuda((2048, 64, 56, 56), (200704, 3136, 56, 1), torch.bfloat16)
        buf41 = empty_strided_cuda((2048, 64, 56, 56), (200704, 1, 3584, 64), torch.int8)
        buf51 = empty_strided_cuda((2048, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        buf151 = empty_strided_cuda((2048, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [maxpool, layer1_0_conv1, layer1_0_downsample_0], Original ATen: [aten.max_pool2d_with_indices, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_12.run(buf33, buf40, buf41, buf51, buf151, 131072, 3136, stream=stream0)
        buf42 = empty_strided_cuda((64, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(primals_8, buf42, 4096, stream=stream0)
        del primals_8
        buf44 = empty_strided_cuda((2048, 64, 18, 18), (20736, 1, 1152, 64), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_14.run(buf40, buf44, 131072, 324, stream=stream0)
        buf45 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf46 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf47 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf40, (802816, 512), (512, 1), 0), buf45, buf46, buf47, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, buf42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf55 = reinterpret_tensor(buf51, (2048, 64, 3136), (200704, 3136, 1), 0); del buf51  # reuse
        buf61 = empty_strided_cuda((2048, 64, 3136), (200704, 3136, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(buf52, buf55, buf61, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf55, buf53, buf54, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf58 = buf22; del buf22  # reuse
        buf62 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16.run(buf54, buf53, buf58, buf62, 64, stream=stream0)
        buf59 = buf55; del buf55  # reuse
        buf60 = reinterpret_tensor(buf52, (2048, 64, 3136), (200704, 3136, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf61, buf62, buf58, primals_10, primals_11, buf59, buf60, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del buf61
        del primals_11
        buf65 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf66 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf67 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf60, (802816, 512), (512, 1), 0), buf65, buf66, buf67, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf71 = empty_strided_cuda((2048, 64, 56, 56), (200704, 3136, 56, 1), torch.int8)
        # Topologically Sorted Source Nodes: [layer1_0_relu], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_17.run(buf71, 411041792, stream=stream0)
        buf72 = reinterpret_tensor(buf60, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf60  # reuse
        buf73 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        buf109 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(buf71, buf73, buf109, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf59, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf72, reinterpret_tensor(buf73, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf76 = empty_strided_cuda((802816, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer1_0_relu], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_19.run(buf76, 12845056, stream=stream0)
        buf77 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        buf112 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_20.run(buf76, buf77, buf112, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf73, (802816, 512), (512, 1), 0), reinterpret_tensor(buf77, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf79 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(primals_14, buf79, 4096, 9, stream=stream0)
        del primals_14
        buf81 = empty_strided_cuda((2048, 64, 18, 18), (20736, 1, 1152, 64), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_14.run(buf72, buf81, 131072, 324, stream=stream0)
        buf82 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf83 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf84 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf72, (802816, 512), (512, 1), 0), buf82, buf83, buf84, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf88 = reinterpret_tensor(buf59, (2048, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf72, buf88, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_conv2], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf92 = reinterpret_tensor(buf88, (2048, 64, 3136), (200704, 3136, 1), 0); del buf88  # reuse
        buf98 = reinterpret_tensor(buf72, (2048, 64, 3136), (200704, 3136, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(buf89, buf92, buf98, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf92, buf90, buf91, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf95 = buf62; del buf62  # reuse
        buf99 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16.run(buf91, buf90, buf95, buf99, 64, stream=stream0)
        buf96 = buf92; del buf92  # reuse
        buf97 = reinterpret_tensor(buf89, (2048, 64, 3136), (200704, 3136, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf98, buf99, buf95, primals_16, primals_17, buf96, buf97, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del buf98
        del primals_17
        buf102 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf103 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf104 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf97, (802816, 512), (512, 1), 0), buf102, buf103, buf104, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf108 = reinterpret_tensor(buf97, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf96, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf108, reinterpret_tensor(buf109, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf109, (802816, 512), (512, 1), 0), reinterpret_tensor(buf112, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf114 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_23.run(primals_20, buf114, 16384, stream=stream0)
        del primals_20
        buf116 = empty_strided_cuda((2048, 64, 18, 18), (20736, 1, 1152, 64), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_14.run(buf108, buf116, 131072, 324, stream=stream0)
        buf117 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf118 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf119 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf108, (802816, 512), (512, 1), 0), buf117, buf118, buf119, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf123 = reinterpret_tensor(buf96, (2048, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf108, buf123, 131072, 3136, stream=stream0)
        del buf108
        # Topologically Sorted Source Nodes: [layer1_0_conv3], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, buf114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (2048, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        del buf123
        buf125 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_24.run(buf125, 256, stream=stream0)
        buf126 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf127 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf153 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf154 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_25.run(buf125, buf126, buf127, buf153, buf154, 256, stream=stream0)
        buf128 = reinterpret_tensor(buf19, (2048, 256, 3136), (802816, 3136, 1), 0); del buf19  # reuse
        buf134 = reinterpret_tensor(buf31, (2048, 256, 3136), (802816, 3136, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf124, buf128, buf134, 524288, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_7.run(buf128, buf126, buf127, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        buf131 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf135 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_27.run(buf127, buf126, buf131, buf135, 256, stream=stream0)
        buf132 = buf128; del buf128  # reuse
        buf133 = reinterpret_tensor(buf124, (2048, 256, 3136), (802816, 3136, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_8.run(buf134, buf135, buf131, primals_22, primals_23, buf132, buf133, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        del primals_23
        buf138 = empty_strided_cuda((3211264, 32), (32, 1), torch.int32)
        buf139 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        buf140 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf133, (3211264, 512), (512, 1), 0), buf138, buf139, buf140, 512, 1, 32, 1, 2, 16, 32, 3, 3211264, 1, 1, stream=stream0)
        buf144 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_downsample_0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_23.run(primals_26, buf144, 16384, stream=stream0)
        del primals_26
        buf145 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf146 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf147 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf40, (802816, 512), (512, 1), 0), buf145, buf146, buf147, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_0_downsample_0], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, buf144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (2048, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf155 = buf133; del buf133  # reuse
        buf161 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf152, buf155, buf161, 524288, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_7.run(buf155, buf153, buf154, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        buf158 = buf135; del buf135  # reuse
        buf162 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_downsample_1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_27.run(buf154, buf153, buf158, buf162, 256, stream=stream0)
        buf159 = buf155; del buf155  # reuse
        buf160 = reinterpret_tensor(buf152, (2048, 256, 3136), (802816, 3136, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_8.run(buf161, buf162, buf158, primals_28, primals_29, buf159, buf160, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        del primals_29
        buf165 = empty_strided_cuda((3211264, 32), (32, 1), torch.int32)
        buf166 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        buf167 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf160, (3211264, 512), (512, 1), 0), buf165, buf166, buf167, 512, 1, 32, 1, 2, 16, 32, 3, 3211264, 1, 1, stream=stream0)
        buf171 = reinterpret_tensor(buf34, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_relu_2], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(buf171, 1644167168, stream=stream0)
        buf172 = reinterpret_tensor(buf160, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf160  # reuse
        buf173 = empty_strided_cuda((1644167168, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf171, buf173, 1644167168, stream=stream0)
        buf174 = reinterpret_tensor(buf161, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf132, buf159, buf174, 1644167168, stream=stream0)
        del buf132
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf174, buf172, reinterpret_tensor(buf173, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), 1644167168, 1024, 1605632, 1, 1, stream=stream0)
        buf177 = empty_strided_cuda((51380224, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf37, buf177, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf173, (3211264, 512), (512, 1), 0), reinterpret_tensor(buf177, (3211264, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 3211264, 1, 1, stream=stream0)
        buf179 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_23.run(primals_32, buf179, 16384, stream=stream0)
        del primals_32
        buf181 = empty_strided_cuda((2048, 256, 18, 18), (82944, 1, 4608, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_30.run(buf172, buf181, 524288, 324, stream=stream0)
        buf182 = empty_strided_cuda((3211264, 32), (32, 1), torch.int32)
        buf183 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        buf184 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf172, (3211264, 512), (512, 1), 0), buf182, buf183, buf184, 512, 1, 32, 1, 2, 16, 32, 3, 3211264, 1, 1, stream=stream0)
        buf188 = reinterpret_tensor(buf174, (2048, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf172, buf188, 524288, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, buf179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf190 = buf99; del buf99  # reuse
        buf191 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf225 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf226 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf12, buf190, buf191, buf225, buf226, 64, stream=stream0)
        buf192 = reinterpret_tensor(buf151, (2048, 64, 3136), (200704, 3136, 1), 0); del buf151  # reuse
        buf198 = reinterpret_tensor(buf40, (2048, 64, 3136), (200704, 3136, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(buf189, buf192, buf198, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf192, buf190, buf191, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf195 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf199 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16.run(buf191, buf190, buf195, buf199, 64, stream=stream0)
        buf196 = buf192; del buf192  # reuse
        buf197 = reinterpret_tensor(buf189, (2048, 64, 3136), (200704, 3136, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf198, buf199, buf195, primals_34, primals_35, buf196, buf197, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del buf198
        del primals_35
        buf202 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf203 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf204 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf197, (802816, 512), (512, 1), 0), buf202, buf203, buf204, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf208 = reinterpret_tensor(buf197, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf197  # reuse
        buf209 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        buf244 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(buf71, buf209, buf244, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf196, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf208, reinterpret_tensor(buf209, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf212 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        buf247 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_20.run(buf76, buf212, buf247, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf209, (802816, 512), (512, 1), 0), reinterpret_tensor(buf212, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf214 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_1_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(primals_38, buf214, 4096, 9, stream=stream0)
        del primals_38
        buf216 = empty_strided_cuda((2048, 64, 18, 18), (20736, 1, 1152, 64), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer1_1_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_14.run(buf208, buf216, 131072, 324, stream=stream0)
        buf217 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf218 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf219 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf208, (802816, 512), (512, 1), 0), buf217, buf218, buf219, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf223 = reinterpret_tensor(buf196, (2048, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf208, buf223, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_1_conv2], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, buf214, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf227 = reinterpret_tensor(buf223, (2048, 64, 3136), (200704, 3136, 1), 0); del buf223  # reuse
        buf233 = reinterpret_tensor(buf208, (2048, 64, 3136), (200704, 3136, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(buf224, buf227, buf233, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf227, buf225, buf226, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf230 = buf199; del buf199  # reuse
        buf234 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16.run(buf226, buf225, buf230, buf234, 64, stream=stream0)
        buf231 = buf227; del buf227  # reuse
        buf232 = reinterpret_tensor(buf224, (2048, 64, 3136), (200704, 3136, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf233, buf234, buf230, primals_40, primals_41, buf231, buf232, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del buf233
        del primals_41
        buf237 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf238 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf239 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf232, (802816, 512), (512, 1), 0), buf237, buf238, buf239, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf243 = reinterpret_tensor(buf232, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf231, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf243, reinterpret_tensor(buf244, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf244, (802816, 512), (512, 1), 0), reinterpret_tensor(buf247, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf249 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_1_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_23.run(primals_44, buf249, 16384, stream=stream0)
        del primals_44
        buf251 = empty_strided_cuda((2048, 64, 18, 18), (20736, 1, 1152, 64), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer1_1_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_14.run(buf243, buf251, 131072, 324, stream=stream0)
        buf252 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf253 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf254 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf243, (802816, 512), (512, 1), 0), buf252, buf253, buf254, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf258 = reinterpret_tensor(buf231, (2048, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf243, buf258, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_1_conv3], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, buf249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (2048, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf260 = buf162; del buf162  # reuse
        buf261 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_33.run(buf125, buf260, buf261, 256, stream=stream0)
        buf262 = reinterpret_tensor(buf188, (2048, 256, 3136), (802816, 3136, 1), 0); del buf188  # reuse
        buf268 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf259, buf262, buf268, 524288, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_7.run(buf262, buf260, buf261, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        buf265 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf269 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_27.run(buf261, buf260, buf265, buf269, 256, stream=stream0)
        buf266 = buf262; del buf262  # reuse
        buf267 = reinterpret_tensor(buf259, (2048, 256, 3136), (802816, 3136, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_8.run(buf268, buf269, buf265, primals_46, primals_47, buf266, buf267, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        del primals_47
        buf272 = empty_strided_cuda((3211264, 32), (32, 1), torch.int32)
        buf273 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        buf274 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf267, (3211264, 512), (512, 1), 0), buf272, buf273, buf274, 512, 1, 32, 1, 2, 16, 32, 3, 3211264, 1, 1, stream=stream0)
        buf278 = reinterpret_tensor(buf267, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf267  # reuse
        buf279 = empty_strided_cuda((1644167168, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf171, buf279, 1644167168, stream=stream0)
        buf280 = reinterpret_tensor(buf268, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf266, buf172, buf280, 1644167168, stream=stream0)
        del buf172
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf280, buf278, reinterpret_tensor(buf279, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), 1644167168, 1024, 1605632, 1, 1, stream=stream0)
        buf283 = empty_strided_cuda((51380224, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf37, buf283, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf279, (3211264, 512), (512, 1), 0), reinterpret_tensor(buf283, (3211264, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 3211264, 1, 1, stream=stream0)
        buf285 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_23.run(primals_50, buf285, 16384, stream=stream0)
        del primals_50
        buf287 = empty_strided_cuda((2048, 256, 18, 18), (82944, 1, 4608, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_30.run(buf278, buf287, 524288, 324, stream=stream0)
        buf288 = empty_strided_cuda((3211264, 32), (32, 1), torch.int32)
        buf289 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        buf290 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf278, (3211264, 512), (512, 1), 0), buf288, buf289, buf290, 512, 1, 32, 1, 2, 16, 32, 3, 3211264, 1, 1, stream=stream0)
        buf294 = reinterpret_tensor(buf280, (2048, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf278, buf294, 524288, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, buf285, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf296 = buf234; del buf234  # reuse
        buf297 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf331 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf12, buf296, buf297, buf331, 64, stream=stream0)
        buf298 = reinterpret_tensor(buf258, (2048, 64, 3136), (200704, 3136, 1), 0); del buf258  # reuse
        buf304 = reinterpret_tensor(buf243, (2048, 64, 3136), (200704, 3136, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(buf295, buf298, buf304, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf298, buf296, buf297, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf301 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf305 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16.run(buf297, buf296, buf301, buf305, 64, stream=stream0)
        buf302 = buf298; del buf298  # reuse
        buf303 = reinterpret_tensor(buf295, (2048, 64, 3136), (200704, 3136, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf304, buf305, buf301, primals_52, primals_53, buf302, buf303, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del buf304
        del primals_53
        buf308 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf309 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf310 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf303, (802816, 512), (512, 1), 0), buf308, buf309, buf310, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf314 = reinterpret_tensor(buf303, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf303  # reuse
        buf315 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf71, buf315, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf302, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf314, reinterpret_tensor(buf315, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf318 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        buf351 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_20.run(buf76, buf318, buf351, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf315, (802816, 512), (512, 1), 0), reinterpret_tensor(buf318, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf320 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_2_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_21.run(primals_56, buf320, 4096, 9, stream=stream0)
        del primals_56
        buf322 = empty_strided_cuda((2048, 64, 18, 18), (20736, 1, 1152, 64), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer1_2_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_14.run(buf314, buf322, 131072, 324, stream=stream0)
        buf323 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf324 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf325 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf314, (802816, 512), (512, 1), 0), buf323, buf324, buf325, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf329 = reinterpret_tensor(buf302, (2048, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf314, buf329, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_2_conv2], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, buf320, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf332 = reinterpret_tensor(buf329, (2048, 64, 3136), (200704, 3136, 1), 0); del buf329  # reuse
        buf338 = reinterpret_tensor(buf314, (2048, 64, 3136), (200704, 3136, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(buf330, buf332, buf338, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_5.run(buf332, buf12, buf331, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf335 = buf305; del buf305  # reuse
        buf339 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_16.run(buf331, buf12, buf335, buf339, 64, stream=stream0)
        buf336 = buf332; del buf332  # reuse
        buf337 = reinterpret_tensor(buf330, (2048, 64, 3136), (200704, 3136, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_6.run(buf338, buf339, buf335, primals_58, primals_59, buf336, buf337, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del buf338
        del buf339
        del primals_59
        buf342 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf343 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf344 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf337, (802816, 512), (512, 1), 0), buf342, buf343, buf344, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf348 = reinterpret_tensor(buf337, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf336, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf348, buf71, 411041792, 1024, 401408, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf71, (802816, 512), (512, 1), 0), reinterpret_tensor(buf351, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf353 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_2_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_23.run(primals_62, buf353, 16384, stream=stream0)
        del primals_62
        buf355 = empty_strided_cuda((2048, 64, 18, 18), (20736, 1, 1152, 64), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer1_2_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_14.run(buf348, buf355, 131072, 324, stream=stream0)
        buf356 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf357 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf358 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf348, (802816, 512), (512, 1), 0), buf356, buf357, buf358, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf362 = reinterpret_tensor(buf336, (2048, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_22.run(buf348, buf362, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer1_2_conv3], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, buf353, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (2048, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf364 = buf269; del buf269  # reuse
        buf365 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_33.run(buf125, buf364, buf365, 256, stream=stream0)
        buf366 = reinterpret_tensor(buf294, (2048, 256, 3136), (802816, 3136, 1), 0); del buf294  # reuse
        buf372 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf363, buf366, buf372, 524288, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_7.run(buf366, buf364, buf365, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        buf369 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf373 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_27.run(buf365, buf364, buf369, buf373, 256, stream=stream0)
        buf370 = buf366; del buf366  # reuse
        buf371 = reinterpret_tensor(buf363, (2048, 256, 3136), (802816, 3136, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_8.run(buf372, buf373, buf369, primals_64, primals_65, buf370, buf371, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        del primals_65
        buf376 = empty_strided_cuda((3211264, 32), (32, 1), torch.int32)
        buf377 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        buf378 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf371, (3211264, 512), (512, 1), 0), buf376, buf377, buf378, 512, 1, 32, 1, 2, 16, 32, 3, 3211264, 1, 1, stream=stream0)
        buf382 = reinterpret_tensor(buf371, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf371  # reuse
        buf383 = reinterpret_tensor(buf372, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf370, buf278, buf383, 1644167168, stream=stream0)
        del buf278
        del buf370
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf383, buf382, buf171, 1644167168, 1024, 1605632, 1, 1, stream=stream0)
        del buf173
        del buf279
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf171, (3211264, 512), (512, 1), 0), buf37, 512, 1, 16, 1, 1, 32, 16, 3211264, 1, 1, stream=stream0)
        del buf171
        buf387 = empty_strided_cuda((128, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_36.run(primals_68, buf387, 32768, stream=stream0)
        del primals_68
        buf389 = empty_strided_cuda((2048, 256, 18, 18), (82944, 1, 4608, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_30.run(buf382, buf389, 524288, 324, stream=stream0)
        buf390 = empty_strided_cuda((3211264, 32), (32, 1), torch.int32)
        buf391 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        buf392 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf382, (3211264, 512), (512, 1), 0), buf390, buf391, buf392, 512, 1, 32, 1, 2, 16, 32, 3, 3211264, 1, 1, stream=stream0)
        buf396 = reinterpret_tensor(buf383, (2048, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf382, buf396, 524288, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, buf387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (2048, 128, 56, 56), (401408, 1, 7168, 128), 'torch.ops.aten.convolution.default')
        buf398 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_37.run(buf398, 128, stream=stream0)
        buf399 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf400 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf435 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf436 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf398, buf399, buf400, buf435, buf436, 128, stream=stream0)
        buf401 = empty_strided_cuda((2048, 128, 3136), (401408, 3136, 1), torch.bfloat16)
        buf407 = empty_strided_cuda((2048, 128, 3136), (401408, 3136, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_39.run(buf397, buf401, buf407, 262144, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_9.run(buf401, buf399, buf400, 6422528, 3136, 401408, 3136, 1024, 128, 6272, 1, stream=stream0)
        buf404 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf408 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer2_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_40.run(buf400, buf399, buf404, buf408, 128, stream=stream0)
        buf405 = buf401; del buf401  # reuse
        buf406 = reinterpret_tensor(buf397, (2048, 128, 3136), (401408, 3136, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_10.run(buf407, buf408, buf404, primals_70, primals_71, buf405, buf406, 6422528, 3136, 401408, 3136, 1024, 128, 6272, 1, stream=stream0)
        del buf407
        del primals_71
        buf411 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf412 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf413 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf406, (1605632, 512), (512, 1), 0), buf411, buf412, buf413, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf417 = reinterpret_tensor(buf406, (2048, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf406  # reuse
        buf418 = empty_strided_cuda((2048, 128, 56, 56), (401408, 3136, 56, 1), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_41.run(buf418, 822083584, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf405, (2048, 128, 56, 56), (401408, 3136, 56, 1), 0), buf417, buf418, 822083584, 1024, 802816, 1, 1, stream=stream0)
        buf421 = empty_strided_cuda((1605632, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer2_0_relu], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_42.run(buf421, 25690112, stream=stream0)
        buf422 = empty_strided_cuda((25690112, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf421, buf422, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf418, (1605632, 512), (512, 1), 0), reinterpret_tensor(buf422, (1605632, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        buf424 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_74, buf424, 16384, 9, stream=stream0)
        del primals_74
        buf426 = empty_strided_cuda((2048, 128, 18, 18), (41472, 1, 2304, 128), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_45.run(buf417, buf426, 262144, 324, stream=stream0)
        buf427 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf428 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf429 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf417, (1605632, 512), (512, 1), 0), buf427, buf428, buf429, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf433 = reinterpret_tensor(buf405, (2048, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_46.run(buf417, buf433, 262144, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_0_conv2], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, buf424, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf437 = empty_strided_cuda((2048, 128, 784), (100352, 784, 1), torch.bfloat16)
        buf443 = empty_strided_cuda((2048, 128, 784), (100352, 784, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf434, buf437, buf443, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf437, buf435, buf436, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf440 = buf408; del buf408  # reuse
        buf444 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48.run(buf436, buf435, buf440, buf444, 128, stream=stream0)
        buf441 = buf437; del buf437  # reuse
        buf442 = reinterpret_tensor(buf434, (2048, 128, 784), (100352, 784, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf443, buf444, buf440, primals_76, primals_77, buf441, buf442, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del buf443
        del primals_77
        buf447 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf448 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf449 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf442, (401408, 512), (512, 1), 0), buf447, buf448, buf449, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf453 = empty_strided_cuda((2048, 128, 28, 28), (100352, 784, 28, 1), torch.int8)
        # Topologically Sorted Source Nodes: [layer2_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_49.run(buf453, 205520896, stream=stream0)
        buf454 = reinterpret_tensor(buf442, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf442  # reuse
        buf455 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_50.run(buf453, buf455, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf441, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf454, reinterpret_tensor(buf455, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf458 = empty_strided_cuda((401408, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer2_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_51.run(buf458, 6422528, stream=stream0)
        buf459 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_52.run(buf458, buf459, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf455, (401408, 512), (512, 1), 0), reinterpret_tensor(buf459, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf461 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_53.run(primals_80, buf461, 65536, stream=stream0)
        del primals_80
        buf463 = empty_strided_cuda((2048, 128, 9, 9), (10368, 1, 1152, 128), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_54.run(buf454, buf463, 262144, 81, stream=stream0)
        buf464 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf465 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf466 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf454, (401408, 512), (512, 1), 0), buf464, buf465, buf466, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf470 = reinterpret_tensor(buf441, (2048, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf454, buf470, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_0_conv3], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, buf461, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf472 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_56.run(buf472, 512, stream=stream0)
        buf473 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf474 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf500 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf501 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf472, buf473, buf474, buf500, buf501, 512, stream=stream0)
        buf475 = reinterpret_tensor(buf433, (2048, 512, 784), (401408, 784, 1), 0); del buf433  # reuse
        buf481 = reinterpret_tensor(buf417, (2048, 512, 784), (401408, 784, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_58.run(buf471, buf475, buf481, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf475, buf473, buf474, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf478 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf482 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_59.run(buf474, buf473, buf478, buf482, 512, stream=stream0)
        buf479 = buf475; del buf475  # reuse
        buf480 = reinterpret_tensor(buf471, (2048, 512, 784), (401408, 784, 1), 0); del buf471  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf481, buf482, buf478, primals_82, primals_83, buf479, buf480, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del primals_83
        buf485 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf486 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf487 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf480, (1605632, 512), (512, 1), 0), buf485, buf486, buf487, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf491 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_60.run(primals_86, buf491, 131072, stream=stream0)
        del primals_86
        buf492 = empty_strided_cuda((3211264, 32), (32, 1), torch.int32)
        buf493 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        buf494 = empty_strided_cuda((3211264, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf382, (3211264, 512), (512, 1), 0), buf492, buf493, buf494, 512, 1, 32, 1, 2, 16, 32, 3, 3211264, 1, 1, stream=stream0)
        buf498 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_31.run(buf382, buf498, 524288, 3136, stream=stream0)
        del buf382
        # Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, buf491, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        del buf498
        buf502 = buf480; del buf480  # reuse
        buf508 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_58.run(buf499, buf502, buf508, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf502, buf500, buf501, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf505 = buf482; del buf482  # reuse
        buf509 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_downsample_1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_59.run(buf501, buf500, buf505, buf509, 512, stream=stream0)
        buf506 = buf502; del buf502  # reuse
        buf507 = reinterpret_tensor(buf499, (2048, 512, 784), (401408, 784, 1), 0); del buf499  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf508, buf509, buf505, primals_88, primals_89, buf506, buf507, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del primals_89
        buf512 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf513 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf514 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf507, (1605632, 512), (512, 1), 0), buf512, buf513, buf514, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf518 = reinterpret_tensor(buf418, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_relu_2], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_41.run(buf518, 822083584, stream=stream0)
        buf519 = reinterpret_tensor(buf507, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf507  # reuse
        buf520 = empty_strided_cuda((822083584, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_61.run(buf518, buf520, 822083584, stream=stream0)
        buf521 = reinterpret_tensor(buf508, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_62.run(buf479, buf506, buf521, 822083584, stream=stream0)
        del buf479
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf521, buf519, reinterpret_tensor(buf520, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), 822083584, 1024, 802816, 1, 1, stream=stream0)
        buf524 = empty_strided_cuda((25690112, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf421, buf524, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf520, (1605632, 512), (512, 1), 0), reinterpret_tensor(buf524, (1605632, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        buf526 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_53.run(primals_92, buf526, 65536, stream=stream0)
        del primals_92
        buf528 = empty_strided_cuda((2048, 512, 9, 9), (41472, 1, 4608, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_63.run(buf519, buf528, 1048576, 81, stream=stream0)
        buf529 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf530 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf531 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf519, (1605632, 512), (512, 1), 0), buf529, buf530, buf531, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf535 = reinterpret_tensor(buf521, (2048, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf521  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_64.run(buf519, buf535, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, buf526, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf537 = buf444; del buf444  # reuse
        buf538 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf572 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf573 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf398, buf537, buf538, buf572, buf573, 128, stream=stream0)
        buf539 = reinterpret_tensor(buf470, (2048, 128, 784), (100352, 784, 1), 0); del buf470  # reuse
        buf545 = reinterpret_tensor(buf454, (2048, 128, 784), (100352, 784, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf536, buf539, buf545, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf539, buf537, buf538, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf542 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf546 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48.run(buf538, buf537, buf542, buf546, 128, stream=stream0)
        buf543 = buf539; del buf539  # reuse
        buf544 = reinterpret_tensor(buf536, (2048, 128, 784), (100352, 784, 1), 0); del buf536  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf545, buf546, buf542, primals_94, primals_95, buf543, buf544, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del buf545
        del primals_95
        buf549 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf550 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf551 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf544, (401408, 512), (512, 1), 0), buf549, buf550, buf551, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf555 = reinterpret_tensor(buf544, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf544  # reuse
        buf556 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        buf591 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_65.run(buf453, buf556, buf591, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf543, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf555, reinterpret_tensor(buf556, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf559 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        buf594 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_66.run(buf458, buf559, buf594, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf556, (401408, 512), (512, 1), 0), reinterpret_tensor(buf559, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf561 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_1_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_98, buf561, 16384, 9, stream=stream0)
        del primals_98
        buf563 = empty_strided_cuda((2048, 128, 9, 9), (10368, 1, 1152, 128), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_1_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_54.run(buf555, buf563, 262144, 81, stream=stream0)
        buf564 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf565 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf566 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf555, (401408, 512), (512, 1), 0), buf564, buf565, buf566, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf570 = reinterpret_tensor(buf543, (2048, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf555, buf570, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_1_conv2], Original ATen: [aten.convolution]
        buf571 = extern_kernels.convolution(buf570, buf561, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf574 = reinterpret_tensor(buf570, (2048, 128, 784), (100352, 784, 1), 0); del buf570  # reuse
        buf580 = reinterpret_tensor(buf555, (2048, 128, 784), (100352, 784, 1), 0); del buf555  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf571, buf574, buf580, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf574, buf572, buf573, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf577 = buf546; del buf546  # reuse
        buf581 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48.run(buf573, buf572, buf577, buf581, 128, stream=stream0)
        buf578 = buf574; del buf574  # reuse
        buf579 = reinterpret_tensor(buf571, (2048, 128, 784), (100352, 784, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf580, buf581, buf577, primals_100, primals_101, buf578, buf579, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del buf580
        del primals_101
        buf584 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf585 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf586 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf579, (401408, 512), (512, 1), 0), buf584, buf585, buf586, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf590 = reinterpret_tensor(buf579, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf579  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf578, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf590, reinterpret_tensor(buf591, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf591, (401408, 512), (512, 1), 0), reinterpret_tensor(buf594, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf596 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_1_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_53.run(primals_104, buf596, 65536, stream=stream0)
        del primals_104
        buf598 = empty_strided_cuda((2048, 128, 9, 9), (10368, 1, 1152, 128), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_1_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_54.run(buf590, buf598, 262144, 81, stream=stream0)
        buf599 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf600 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf601 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf590, (401408, 512), (512, 1), 0), buf599, buf600, buf601, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf605 = reinterpret_tensor(buf578, (2048, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf590, buf605, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_1_conv3], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, buf596, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf607 = buf509; del buf509  # reuse
        buf608 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_67.run(buf472, buf607, buf608, 512, stream=stream0)
        buf609 = reinterpret_tensor(buf535, (2048, 512, 784), (401408, 784, 1), 0); del buf535  # reuse
        buf615 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_58.run(buf606, buf609, buf615, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf609, buf607, buf608, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf612 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf616 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_59.run(buf608, buf607, buf612, buf616, 512, stream=stream0)
        buf613 = buf609; del buf609  # reuse
        buf614 = reinterpret_tensor(buf606, (2048, 512, 784), (401408, 784, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf615, buf616, buf612, primals_106, primals_107, buf613, buf614, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del primals_107
        buf619 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf620 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf621 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf614, (1605632, 512), (512, 1), 0), buf619, buf620, buf621, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf625 = reinterpret_tensor(buf614, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf614  # reuse
        buf626 = empty_strided_cuda((822083584, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_61.run(buf518, buf626, 822083584, stream=stream0)
        buf627 = reinterpret_tensor(buf615, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf615  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_62.run(buf613, buf519, buf627, 822083584, stream=stream0)
        del buf519
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf627, buf625, reinterpret_tensor(buf626, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), 822083584, 1024, 802816, 1, 1, stream=stream0)
        buf630 = empty_strided_cuda((25690112, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf421, buf630, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf626, (1605632, 512), (512, 1), 0), reinterpret_tensor(buf630, (1605632, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        buf632 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_53.run(primals_110, buf632, 65536, stream=stream0)
        del primals_110
        buf634 = empty_strided_cuda((2048, 512, 9, 9), (41472, 1, 4608, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_63.run(buf625, buf634, 1048576, 81, stream=stream0)
        buf635 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf636 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf637 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf625, (1605632, 512), (512, 1), 0), buf635, buf636, buf637, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf641 = reinterpret_tensor(buf627, (2048, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf627  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_64.run(buf625, buf641, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten.convolution]
        buf642 = extern_kernels.convolution(buf641, buf632, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf642, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf643 = buf581; del buf581  # reuse
        buf644 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf678 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf679 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf398, buf643, buf644, buf678, buf679, 128, stream=stream0)
        buf645 = reinterpret_tensor(buf605, (2048, 128, 784), (100352, 784, 1), 0); del buf605  # reuse
        buf651 = reinterpret_tensor(buf590, (2048, 128, 784), (100352, 784, 1), 0); del buf590  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf642, buf645, buf651, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf645, buf643, buf644, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf648 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf652 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48.run(buf644, buf643, buf648, buf652, 128, stream=stream0)
        buf649 = buf645; del buf645  # reuse
        buf650 = reinterpret_tensor(buf642, (2048, 128, 784), (100352, 784, 1), 0); del buf642  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf651, buf652, buf648, primals_112, primals_113, buf649, buf650, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del buf651
        del primals_113
        buf655 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf656 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf657 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf650, (401408, 512), (512, 1), 0), buf655, buf656, buf657, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf661 = reinterpret_tensor(buf650, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf650  # reuse
        buf662 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        buf697 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_65.run(buf453, buf662, buf697, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf649, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf661, reinterpret_tensor(buf662, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf665 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        buf700 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_66.run(buf458, buf665, buf700, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf662, (401408, 512), (512, 1), 0), reinterpret_tensor(buf665, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf667 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_2_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_116, buf667, 16384, 9, stream=stream0)
        del primals_116
        buf669 = empty_strided_cuda((2048, 128, 9, 9), (10368, 1, 1152, 128), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_2_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_54.run(buf661, buf669, 262144, 81, stream=stream0)
        buf670 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf671 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf672 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf661, (401408, 512), (512, 1), 0), buf670, buf671, buf672, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf676 = reinterpret_tensor(buf649, (2048, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf649  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf661, buf676, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_2_conv2], Original ATen: [aten.convolution]
        buf677 = extern_kernels.convolution(buf676, buf667, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf677, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf680 = reinterpret_tensor(buf676, (2048, 128, 784), (100352, 784, 1), 0); del buf676  # reuse
        buf686 = reinterpret_tensor(buf661, (2048, 128, 784), (100352, 784, 1), 0); del buf661  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf677, buf680, buf686, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf680, buf678, buf679, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf683 = buf652; del buf652  # reuse
        buf687 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48.run(buf679, buf678, buf683, buf687, 128, stream=stream0)
        buf684 = buf680; del buf680  # reuse
        buf685 = reinterpret_tensor(buf677, (2048, 128, 784), (100352, 784, 1), 0); del buf677  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf686, buf687, buf683, primals_118, primals_119, buf684, buf685, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del buf686
        del primals_119
        buf690 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf691 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf692 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf685, (401408, 512), (512, 1), 0), buf690, buf691, buf692, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf696 = reinterpret_tensor(buf685, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf685  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf684, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf696, reinterpret_tensor(buf697, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf697, (401408, 512), (512, 1), 0), reinterpret_tensor(buf700, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf702 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_2_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_53.run(primals_122, buf702, 65536, stream=stream0)
        del primals_122
        buf704 = empty_strided_cuda((2048, 128, 9, 9), (10368, 1, 1152, 128), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_2_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_54.run(buf696, buf704, 262144, 81, stream=stream0)
        buf705 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf706 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf707 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf696, (401408, 512), (512, 1), 0), buf705, buf706, buf707, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf711 = reinterpret_tensor(buf684, (2048, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf684  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf696, buf711, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_2_conv3], Original ATen: [aten.convolution]
        buf712 = extern_kernels.convolution(buf711, buf702, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf712, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf713 = buf616; del buf616  # reuse
        buf714 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_67.run(buf472, buf713, buf714, 512, stream=stream0)
        buf715 = reinterpret_tensor(buf641, (2048, 512, 784), (401408, 784, 1), 0); del buf641  # reuse
        buf721 = buf613; del buf613  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_58.run(buf712, buf715, buf721, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf715, buf713, buf714, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf718 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf722 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_59.run(buf714, buf713, buf718, buf722, 512, stream=stream0)
        buf719 = buf715; del buf715  # reuse
        buf720 = reinterpret_tensor(buf712, (2048, 512, 784), (401408, 784, 1), 0); del buf712  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf721, buf722, buf718, primals_124, primals_125, buf719, buf720, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del primals_125
        buf725 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf726 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf727 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf720, (1605632, 512), (512, 1), 0), buf725, buf726, buf727, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf731 = reinterpret_tensor(buf720, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf720  # reuse
        buf732 = empty_strided_cuda((822083584, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_61.run(buf518, buf732, 822083584, stream=stream0)
        buf733 = reinterpret_tensor(buf721, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf721  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_62.run(buf719, buf625, buf733, 822083584, stream=stream0)
        del buf625
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf733, buf731, reinterpret_tensor(buf732, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), 822083584, 1024, 802816, 1, 1, stream=stream0)
        buf736 = empty_strided_cuda((25690112, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf421, buf736, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf732, (1605632, 512), (512, 1), 0), reinterpret_tensor(buf736, (1605632, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        buf738 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_53.run(primals_128, buf738, 65536, stream=stream0)
        del primals_128
        buf740 = empty_strided_cuda((2048, 512, 9, 9), (41472, 1, 4608, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_63.run(buf731, buf740, 1048576, 81, stream=stream0)
        buf741 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf742 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf743 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf731, (1605632, 512), (512, 1), 0), buf741, buf742, buf743, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf747 = reinterpret_tensor(buf733, (2048, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf733  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_64.run(buf731, buf747, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten.convolution]
        buf748 = extern_kernels.convolution(buf747, buf738, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf748, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf749 = buf687; del buf687  # reuse
        buf750 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf784 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf398, buf749, buf750, buf784, 128, stream=stream0)
        buf751 = reinterpret_tensor(buf711, (2048, 128, 784), (100352, 784, 1), 0); del buf711  # reuse
        buf757 = reinterpret_tensor(buf696, (2048, 128, 784), (100352, 784, 1), 0); del buf696  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf748, buf751, buf757, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf751, buf749, buf750, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf754 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf758 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48.run(buf750, buf749, buf754, buf758, 128, stream=stream0)
        buf755 = buf751; del buf751  # reuse
        buf756 = reinterpret_tensor(buf748, (2048, 128, 784), (100352, 784, 1), 0); del buf748  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf757, buf758, buf754, primals_130, primals_131, buf755, buf756, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del buf757
        del primals_131
        buf761 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf762 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf763 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf756, (401408, 512), (512, 1), 0), buf761, buf762, buf763, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf767 = reinterpret_tensor(buf756, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf756  # reuse
        buf768 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_50.run(buf453, buf768, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf755, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf767, reinterpret_tensor(buf768, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf771 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        buf804 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_66.run(buf458, buf771, buf804, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf768, (401408, 512), (512, 1), 0), reinterpret_tensor(buf771, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf773 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_3_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_134, buf773, 16384, 9, stream=stream0)
        del primals_134
        buf775 = empty_strided_cuda((2048, 128, 9, 9), (10368, 1, 1152, 128), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_3_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_54.run(buf767, buf775, 262144, 81, stream=stream0)
        buf776 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf777 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf778 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf767, (401408, 512), (512, 1), 0), buf776, buf777, buf778, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf782 = reinterpret_tensor(buf755, (2048, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf755  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf767, buf782, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_3_conv2], Original ATen: [aten.convolution]
        buf783 = extern_kernels.convolution(buf782, buf773, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf783, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf785 = reinterpret_tensor(buf782, (2048, 128, 784), (100352, 784, 1), 0); del buf782  # reuse
        buf791 = reinterpret_tensor(buf767, (2048, 128, 784), (100352, 784, 1), 0); del buf767  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf783, buf785, buf791, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_11.run(buf785, buf398, buf784, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf788 = buf758; del buf758  # reuse
        buf792 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_48.run(buf784, buf398, buf788, buf792, 128, stream=stream0)
        buf789 = buf785; del buf785  # reuse
        buf790 = reinterpret_tensor(buf783, (2048, 128, 784), (100352, 784, 1), 0); del buf783  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_12.run(buf791, buf792, buf788, primals_136, primals_137, buf789, buf790, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del buf791
        del buf792
        del primals_137
        buf795 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf796 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf797 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf790, (401408, 512), (512, 1), 0), buf795, buf796, buf797, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf801 = reinterpret_tensor(buf790, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf790  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf789, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf801, buf453, 205520896, 1024, 200704, 1, 1, stream=stream0)
        del buf455
        del buf556
        del buf591
        del buf662
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf453, (401408, 512), (512, 1), 0), reinterpret_tensor(buf804, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf806 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_3_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_53.run(primals_140, buf806, 65536, stream=stream0)
        del primals_140
        buf808 = empty_strided_cuda((2048, 128, 9, 9), (10368, 1, 1152, 128), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer2_3_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_54.run(buf801, buf808, 262144, 81, stream=stream0)
        buf809 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf810 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf811 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf801, (401408, 512), (512, 1), 0), buf809, buf810, buf811, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf815 = reinterpret_tensor(buf789, (2048, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf789  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf801, buf815, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer2_3_conv3], Original ATen: [aten.convolution]
        buf816 = extern_kernels.convolution(buf815, buf806, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf816, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf817 = buf722; del buf722  # reuse
        buf818 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_67.run(buf472, buf817, buf818, 512, stream=stream0)
        buf819 = reinterpret_tensor(buf747, (2048, 512, 784), (401408, 784, 1), 0); del buf747  # reuse
        buf825 = buf719; del buf719  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_58.run(buf816, buf819, buf825, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_13.run(buf819, buf817, buf818, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf822 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf826 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_59.run(buf818, buf817, buf822, buf826, 512, stream=stream0)
        buf823 = buf819; del buf819  # reuse
        buf824 = reinterpret_tensor(buf816, (2048, 512, 784), (401408, 784, 1), 0); del buf816  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_14.run(buf825, buf826, buf822, primals_142, primals_143, buf823, buf824, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del primals_143
        buf829 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf830 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf831 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf824, (1605632, 512), (512, 1), 0), buf829, buf830, buf831, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf835 = reinterpret_tensor(buf824, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf824  # reuse
        buf836 = reinterpret_tensor(buf825, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf825  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_62.run(buf823, buf731, buf836, 822083584, stream=stream0)
        del buf731
        del buf823
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf836, buf835, buf518, 822083584, 1024, 802816, 1, 1, stream=stream0)
        del buf520
        del buf626
        del buf732
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf518, (1605632, 512), (512, 1), 0), buf421, 512, 1, 16, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        del buf518
        buf840 = empty_strided_cuda((256, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_60.run(primals_146, buf840, 131072, stream=stream0)
        del primals_146
        buf842 = empty_strided_cuda((2048, 512, 9, 9), (41472, 1, 4608, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_63.run(buf835, buf842, 1048576, 81, stream=stream0)
        buf843 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf844 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf845 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf835, (1605632, 512), (512, 1), 0), buf843, buf844, buf845, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf849 = reinterpret_tensor(buf836, (2048, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf836  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_64.run(buf835, buf849, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten.convolution]
        buf850 = extern_kernels.convolution(buf849, buf840, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf850, (2048, 256, 28, 28), (200704, 1, 7168, 256), 'torch.ops.aten.convolution.default')
        buf851 = buf373; del buf373  # reuse
        buf852 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf886 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf887 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_25.run(buf125, buf851, buf852, buf886, buf887, 256, stream=stream0)
        buf853 = reinterpret_tensor(buf362, (2048, 256, 784), (200704, 784, 1), 0); del buf362  # reuse
        buf859 = reinterpret_tensor(buf348, (2048, 256, 784), (200704, 784, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_69.run(buf850, buf853, buf859, 524288, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_15.run(buf853, buf851, buf852, 1605632, 784, 200704, 784, 1024, 256, 1568, 1, stream=stream0)
        buf856 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf860 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer3_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_70.run(buf852, buf851, buf856, buf860, 256, stream=stream0)
        buf857 = buf853; del buf853  # reuse
        buf858 = reinterpret_tensor(buf850, (2048, 256, 784), (200704, 784, 1), 0); del buf850  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_16.run(buf859, buf860, buf856, primals_148, primals_149, buf857, buf858, 1605632, 784, 200704, 784, 1024, 256, 1568, 1, stream=stream0)
        del buf859
        del primals_149
        buf863 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf864 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf865 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf858, (802816, 512), (512, 1), 0), buf863, buf864, buf865, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf869 = reinterpret_tensor(buf858, (2048, 256, 28, 28), (200704, 784, 28, 1), 0); del buf858  # reuse
        buf870 = reinterpret_tensor(buf71, (2048, 256, 28, 28), (200704, 784, 28, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_17.run(buf870, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf857, (2048, 256, 28, 28), (200704, 784, 28, 1), 0), buf869, buf870, 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf873 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_71.run(buf76, buf873, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf870, (802816, 512), (512, 1), 0), reinterpret_tensor(buf873, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf875 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(primals_152, buf875, 65536, 9, stream=stream0)
        del primals_152
        buf877 = empty_strided_cuda((2048, 256, 9, 9), (20736, 1, 2304, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_73.run(buf869, buf877, 524288, 81, stream=stream0)
        buf878 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf879 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf880 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf869, (802816, 512), (512, 1), 0), buf878, buf879, buf880, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf884 = reinterpret_tensor(buf857, (2048, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf857  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_74.run(buf869, buf884, 524288, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_0_conv2], Original ATen: [aten.convolution]
        buf885 = extern_kernels.convolution(buf884, buf875, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf885, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf888 = empty_strided_cuda((2048, 256, 196), (50176, 196, 1), torch.bfloat16)
        buf894 = empty_strided_cuda((2048, 256, 196), (50176, 196, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf885, buf888, buf894, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf888, buf886, buf887, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf891 = buf860; del buf860  # reuse
        buf895 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf887, buf886, buf891, buf895, 256, stream=stream0)
        buf892 = buf888; del buf888  # reuse
        buf893 = reinterpret_tensor(buf885, (2048, 256, 196), (50176, 196, 1), 0); del buf885  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf894, buf895, buf891, primals_154, primals_155, buf892, buf893, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf894
        del primals_155
        buf898 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf899 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf900 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf893, (200704, 512), (512, 1), 0), buf898, buf899, buf900, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf904 = empty_strided_cuda((2048, 256, 14, 14), (50176, 196, 14, 1), torch.int8)
        # Topologically Sorted Source Nodes: [layer3_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_77.run(buf904, 102760448, stream=stream0)
        buf905 = reinterpret_tensor(buf893, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf893  # reuse
        buf906 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf904, buf906, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf892, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf905, reinterpret_tensor(buf906, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf909 = empty_strided_cuda((200704, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer3_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_79.run(buf909, 3211264, stream=stream0)
        buf910 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_80.run(buf909, buf910, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf906, (200704, 512), (512, 1), 0), reinterpret_tensor(buf910, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf912 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_158, buf912, 262144, stream=stream0)
        del primals_158
        buf914 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf905, buf914, 524288, 16, stream=stream0)
        buf915 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf916 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf917 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf905, (200704, 512), (512, 1), 0), buf915, buf916, buf917, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf921 = reinterpret_tensor(buf892, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf892  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf905, buf921, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_0_conv3], Original ATen: [aten.convolution]
        buf922 = extern_kernels.convolution(buf921, buf912, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf922, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf923 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_84.run(buf923, 1024, stream=stream0)
        buf924 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf925 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf951 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf952 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_85.run(buf923, buf924, buf925, buf951, buf952, 1024, stream=stream0)
        buf926 = reinterpret_tensor(buf884, (2048, 1024, 196), (200704, 196, 1), 0); del buf884  # reuse
        buf932 = reinterpret_tensor(buf869, (2048, 1024, 196), (200704, 196, 1), 0); del buf869  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_86.run(buf922, buf926, buf932, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf926, buf924, buf925, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf929 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf933 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87.run(buf925, buf924, buf929, buf933, 1024, stream=stream0)
        buf930 = buf926; del buf926  # reuse
        buf931 = reinterpret_tensor(buf922, (2048, 1024, 196), (200704, 196, 1), 0); del buf922  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf932, buf933, buf929, primals_160, primals_161, buf930, buf931, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_161
        buf936 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf937 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf938 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf931, (802816, 512), (512, 1), 0), buf936, buf937, buf938, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf942 = empty_strided_cuda((1024, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_88.run(primals_164, buf942, 524288, stream=stream0)
        del primals_164
        buf943 = empty_strided_cuda((1605632, 32), (32, 1), torch.int32)
        buf944 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        buf945 = empty_strided_cuda((1605632, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf835, (1605632, 512), (512, 1), 0), buf943, buf944, buf945, 512, 1, 32, 1, 2, 16, 32, 3, 1605632, 1, 1, stream=stream0)
        buf949 = buf849; del buf849  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_64.run(buf835, buf949, 1048576, 784, stream=stream0)
        del buf835
        # Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten.convolution]
        buf950 = extern_kernels.convolution(buf949, buf942, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf950, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        del buf949
        buf953 = buf931; del buf931  # reuse
        buf959 = buf932; del buf932  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_86.run(buf950, buf953, buf959, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf953, buf951, buf952, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf956 = buf933; del buf933  # reuse
        buf960 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_downsample_1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87.run(buf952, buf951, buf956, buf960, 1024, stream=stream0)
        buf957 = buf953; del buf953  # reuse
        buf958 = reinterpret_tensor(buf950, (2048, 1024, 196), (200704, 196, 1), 0); del buf950  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf959, buf960, buf956, primals_166, primals_167, buf957, buf958, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_167
        buf963 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf964 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf965 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf958, (802816, 512), (512, 1), 0), buf963, buf964, buf965, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf969 = reinterpret_tensor(buf870, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf870  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_relu_2], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_17.run(buf969, 411041792, stream=stream0)
        buf970 = reinterpret_tensor(buf958, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf958  # reuse
        buf971 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf969, buf971, 411041792, stream=stream0)
        buf972 = reinterpret_tensor(buf959, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf959  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_89.run(buf930, buf957, buf972, 411041792, stream=stream0)
        del buf930
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf972, buf970, reinterpret_tensor(buf971, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf975 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_71.run(buf76, buf975, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf971, (802816, 512), (512, 1), 0), reinterpret_tensor(buf975, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf977 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_170, buf977, 262144, stream=stream0)
        del primals_170
        buf979 = empty_strided_cuda((2048, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_90.run(buf970, buf979, 2097152, 16, stream=stream0)
        buf980 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf981 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf982 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf970, (802816, 512), (512, 1), 0), buf980, buf981, buf982, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf986 = reinterpret_tensor(buf972, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf972  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_91.run(buf970, buf986, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten.convolution]
        buf987 = extern_kernels.convolution(buf986, buf977, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf987, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf988 = buf895; del buf895  # reuse
        buf989 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1023 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1024 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_25.run(buf125, buf988, buf989, buf1023, buf1024, 256, stream=stream0)
        buf990 = reinterpret_tensor(buf921, (2048, 256, 196), (50176, 196, 1), 0); del buf921  # reuse
        buf996 = reinterpret_tensor(buf905, (2048, 256, 196), (50176, 196, 1), 0); del buf905  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf987, buf990, buf996, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf990, buf988, buf989, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf993 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf997 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf989, buf988, buf993, buf997, 256, stream=stream0)
        buf994 = buf990; del buf990  # reuse
        buf995 = reinterpret_tensor(buf987, (2048, 256, 196), (50176, 196, 1), 0); del buf987  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf996, buf997, buf993, primals_172, primals_173, buf994, buf995, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf996
        del primals_173
        buf1000 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1001 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1002 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf995, (200704, 512), (512, 1), 0), buf1000, buf1001, buf1002, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1006 = reinterpret_tensor(buf995, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf995  # reuse
        buf1007 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf1042 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_92.run(buf904, buf1007, buf1042, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf994, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1006, reinterpret_tensor(buf1007, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf1010 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        buf1045 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_93.run(buf909, buf1010, buf1045, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1007, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1010, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1012 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_1_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(primals_176, buf1012, 65536, 9, stream=stream0)
        del primals_176
        buf1014 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_1_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1006, buf1014, 524288, 16, stream=stream0)
        buf1015 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1016 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1017 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1006, (200704, 512), (512, 1), 0), buf1015, buf1016, buf1017, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1021 = reinterpret_tensor(buf994, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf994  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1006, buf1021, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_1_conv2], Original ATen: [aten.convolution]
        buf1022 = extern_kernels.convolution(buf1021, buf1012, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1022, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1025 = reinterpret_tensor(buf1021, (2048, 256, 196), (50176, 196, 1), 0); del buf1021  # reuse
        buf1031 = reinterpret_tensor(buf1006, (2048, 256, 196), (50176, 196, 1), 0); del buf1006  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf1022, buf1025, buf1031, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1025, buf1023, buf1024, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf1028 = buf997; del buf997  # reuse
        buf1032 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf1024, buf1023, buf1028, buf1032, 256, stream=stream0)
        buf1029 = buf1025; del buf1025  # reuse
        buf1030 = reinterpret_tensor(buf1022, (2048, 256, 196), (50176, 196, 1), 0); del buf1022  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1031, buf1032, buf1028, primals_178, primals_179, buf1029, buf1030, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf1031
        del primals_179
        buf1035 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1036 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1037 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1030, (200704, 512), (512, 1), 0), buf1035, buf1036, buf1037, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1041 = reinterpret_tensor(buf1030, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1030  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1029, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1041, reinterpret_tensor(buf1042, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1042, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1045, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1047 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_1_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_182, buf1047, 262144, stream=stream0)
        del primals_182
        buf1049 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_1_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1041, buf1049, 524288, 16, stream=stream0)
        buf1050 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1051 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1052 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1041, (200704, 512), (512, 1), 0), buf1050, buf1051, buf1052, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1056 = reinterpret_tensor(buf1029, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1029  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1041, buf1056, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_1_conv3], Original ATen: [aten.convolution]
        buf1057 = extern_kernels.convolution(buf1056, buf1047, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1057, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf1058 = buf960; del buf960  # reuse
        buf1059 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_94.run(buf923, buf1058, buf1059, 1024, stream=stream0)
        buf1060 = reinterpret_tensor(buf986, (2048, 1024, 196), (200704, 196, 1), 0); del buf986  # reuse
        buf1066 = buf957; del buf957  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_86.run(buf1057, buf1060, buf1066, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1060, buf1058, buf1059, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf1063 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1067 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87.run(buf1059, buf1058, buf1063, buf1067, 1024, stream=stream0)
        buf1064 = buf1060; del buf1060  # reuse
        buf1065 = reinterpret_tensor(buf1057, (2048, 1024, 196), (200704, 196, 1), 0); del buf1057  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1066, buf1067, buf1063, primals_184, primals_185, buf1064, buf1065, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_185
        buf1070 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1071 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1072 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1065, (802816, 512), (512, 1), 0), buf1070, buf1071, buf1072, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1076 = reinterpret_tensor(buf1065, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1065  # reuse
        buf1077 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf969, buf1077, 411041792, stream=stream0)
        buf1078 = reinterpret_tensor(buf1066, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1066  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_89.run(buf1064, buf970, buf1078, 411041792, stream=stream0)
        del buf1064
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1078, buf1076, reinterpret_tensor(buf1077, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf1081 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_71.run(buf76, buf1081, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1077, (802816, 512), (512, 1), 0), reinterpret_tensor(buf1081, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf1083 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_188, buf1083, 262144, stream=stream0)
        del primals_188
        buf1085 = empty_strided_cuda((2048, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_90.run(buf1076, buf1085, 2097152, 16, stream=stream0)
        buf1086 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1087 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1088 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1076, (802816, 512), (512, 1), 0), buf1086, buf1087, buf1088, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1092 = reinterpret_tensor(buf1078, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1078  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_91.run(buf1076, buf1092, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten.convolution]
        buf1093 = extern_kernels.convolution(buf1092, buf1083, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1093, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1094 = buf1032; del buf1032  # reuse
        buf1095 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1129 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1130 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_25.run(buf125, buf1094, buf1095, buf1129, buf1130, 256, stream=stream0)
        buf1096 = reinterpret_tensor(buf1056, (2048, 256, 196), (50176, 196, 1), 0); del buf1056  # reuse
        buf1102 = reinterpret_tensor(buf1041, (2048, 256, 196), (50176, 196, 1), 0); del buf1041  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf1093, buf1096, buf1102, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1096, buf1094, buf1095, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf1099 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1103 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf1095, buf1094, buf1099, buf1103, 256, stream=stream0)
        buf1100 = buf1096; del buf1096  # reuse
        buf1101 = reinterpret_tensor(buf1093, (2048, 256, 196), (50176, 196, 1), 0); del buf1093  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1102, buf1103, buf1099, primals_190, primals_191, buf1100, buf1101, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf1102
        del primals_191
        buf1106 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1107 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1108 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1101, (200704, 512), (512, 1), 0), buf1106, buf1107, buf1108, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1112 = reinterpret_tensor(buf1101, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1101  # reuse
        buf1113 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf1148 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_92.run(buf904, buf1113, buf1148, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1100, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1112, reinterpret_tensor(buf1113, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf1116 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        buf1151 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_93.run(buf909, buf1116, buf1151, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1113, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1116, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1118 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_2_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(primals_194, buf1118, 65536, 9, stream=stream0)
        del primals_194
        buf1120 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_2_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1112, buf1120, 524288, 16, stream=stream0)
        buf1121 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1122 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1123 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1112, (200704, 512), (512, 1), 0), buf1121, buf1122, buf1123, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1127 = reinterpret_tensor(buf1100, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1100  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1112, buf1127, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_2_conv2], Original ATen: [aten.convolution]
        buf1128 = extern_kernels.convolution(buf1127, buf1118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1128, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1131 = reinterpret_tensor(buf1127, (2048, 256, 196), (50176, 196, 1), 0); del buf1127  # reuse
        buf1137 = reinterpret_tensor(buf1112, (2048, 256, 196), (50176, 196, 1), 0); del buf1112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf1128, buf1131, buf1137, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1131, buf1129, buf1130, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf1134 = buf1103; del buf1103  # reuse
        buf1138 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf1130, buf1129, buf1134, buf1138, 256, stream=stream0)
        buf1135 = buf1131; del buf1131  # reuse
        buf1136 = reinterpret_tensor(buf1128, (2048, 256, 196), (50176, 196, 1), 0); del buf1128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1137, buf1138, buf1134, primals_196, primals_197, buf1135, buf1136, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf1137
        del primals_197
        buf1141 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1142 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1143 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1136, (200704, 512), (512, 1), 0), buf1141, buf1142, buf1143, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1147 = reinterpret_tensor(buf1136, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1135, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1147, reinterpret_tensor(buf1148, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1148, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1151, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1153 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_2_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_200, buf1153, 262144, stream=stream0)
        del primals_200
        buf1155 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_2_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1147, buf1155, 524288, 16, stream=stream0)
        buf1156 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1157 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1158 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1147, (200704, 512), (512, 1), 0), buf1156, buf1157, buf1158, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1162 = reinterpret_tensor(buf1135, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1135  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1147, buf1162, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_2_conv3], Original ATen: [aten.convolution]
        buf1163 = extern_kernels.convolution(buf1162, buf1153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1163, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf1164 = buf1067; del buf1067  # reuse
        buf1165 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_94.run(buf923, buf1164, buf1165, 1024, stream=stream0)
        buf1166 = reinterpret_tensor(buf1092, (2048, 1024, 196), (200704, 196, 1), 0); del buf1092  # reuse
        buf1172 = reinterpret_tensor(buf970, (2048, 1024, 196), (200704, 196, 1), 0); del buf970  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_86.run(buf1163, buf1166, buf1172, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1166, buf1164, buf1165, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf1169 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1173 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87.run(buf1165, buf1164, buf1169, buf1173, 1024, stream=stream0)
        buf1170 = buf1166; del buf1166  # reuse
        buf1171 = reinterpret_tensor(buf1163, (2048, 1024, 196), (200704, 196, 1), 0); del buf1163  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1172, buf1173, buf1169, primals_202, primals_203, buf1170, buf1171, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_203
        buf1176 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1177 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1178 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1171, (802816, 512), (512, 1), 0), buf1176, buf1177, buf1178, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1182 = reinterpret_tensor(buf1171, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1171  # reuse
        buf1183 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf969, buf1183, 411041792, stream=stream0)
        buf1184 = reinterpret_tensor(buf1172, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_89.run(buf1170, buf1076, buf1184, 411041792, stream=stream0)
        del buf1076
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1184, buf1182, reinterpret_tensor(buf1183, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf1187 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_71.run(buf76, buf1187, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1183, (802816, 512), (512, 1), 0), reinterpret_tensor(buf1187, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf1189 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_206, buf1189, 262144, stream=stream0)
        del primals_206
        buf1191 = empty_strided_cuda((2048, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_90.run(buf1182, buf1191, 2097152, 16, stream=stream0)
        buf1192 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1193 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1194 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1182, (802816, 512), (512, 1), 0), buf1192, buf1193, buf1194, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1198 = reinterpret_tensor(buf1184, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1184  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_91.run(buf1182, buf1198, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten.convolution]
        buf1199 = extern_kernels.convolution(buf1198, buf1189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1199, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1200 = buf1138; del buf1138  # reuse
        buf1201 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1235 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1236 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_25.run(buf125, buf1200, buf1201, buf1235, buf1236, 256, stream=stream0)
        buf1202 = reinterpret_tensor(buf1162, (2048, 256, 196), (50176, 196, 1), 0); del buf1162  # reuse
        buf1208 = reinterpret_tensor(buf1147, (2048, 256, 196), (50176, 196, 1), 0); del buf1147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf1199, buf1202, buf1208, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1202, buf1200, buf1201, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf1205 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1209 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf1201, buf1200, buf1205, buf1209, 256, stream=stream0)
        buf1206 = buf1202; del buf1202  # reuse
        buf1207 = reinterpret_tensor(buf1199, (2048, 256, 196), (50176, 196, 1), 0); del buf1199  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1208, buf1209, buf1205, primals_208, primals_209, buf1206, buf1207, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf1208
        del primals_209
        buf1212 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1213 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1214 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1207, (200704, 512), (512, 1), 0), buf1212, buf1213, buf1214, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1218 = reinterpret_tensor(buf1207, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1207  # reuse
        buf1219 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf1254 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_92.run(buf904, buf1219, buf1254, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1206, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1218, reinterpret_tensor(buf1219, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf1222 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        buf1257 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_93.run(buf909, buf1222, buf1257, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1219, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1222, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1224 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_3_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(primals_212, buf1224, 65536, 9, stream=stream0)
        del primals_212
        buf1226 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_3_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1218, buf1226, 524288, 16, stream=stream0)
        buf1227 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1228 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1229 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1218, (200704, 512), (512, 1), 0), buf1227, buf1228, buf1229, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1233 = reinterpret_tensor(buf1206, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1206  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1218, buf1233, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_3_conv2], Original ATen: [aten.convolution]
        buf1234 = extern_kernels.convolution(buf1233, buf1224, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1234, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1237 = reinterpret_tensor(buf1233, (2048, 256, 196), (50176, 196, 1), 0); del buf1233  # reuse
        buf1243 = reinterpret_tensor(buf1218, (2048, 256, 196), (50176, 196, 1), 0); del buf1218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf1234, buf1237, buf1243, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1237, buf1235, buf1236, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf1240 = buf1209; del buf1209  # reuse
        buf1244 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf1236, buf1235, buf1240, buf1244, 256, stream=stream0)
        buf1241 = buf1237; del buf1237  # reuse
        buf1242 = reinterpret_tensor(buf1234, (2048, 256, 196), (50176, 196, 1), 0); del buf1234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1243, buf1244, buf1240, primals_214, primals_215, buf1241, buf1242, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf1243
        del primals_215
        buf1247 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1248 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1249 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1242, (200704, 512), (512, 1), 0), buf1247, buf1248, buf1249, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1253 = reinterpret_tensor(buf1242, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1241, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1253, reinterpret_tensor(buf1254, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1254, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1257, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1259 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_3_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_218, buf1259, 262144, stream=stream0)
        del primals_218
        buf1261 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_3_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1253, buf1261, 524288, 16, stream=stream0)
        buf1262 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1263 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1264 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1253, (200704, 512), (512, 1), 0), buf1262, buf1263, buf1264, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1268 = reinterpret_tensor(buf1241, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1241  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1253, buf1268, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_3_conv3], Original ATen: [aten.convolution]
        buf1269 = extern_kernels.convolution(buf1268, buf1259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1269, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf1270 = buf1173; del buf1173  # reuse
        buf1271 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_94.run(buf923, buf1270, buf1271, 1024, stream=stream0)
        buf1272 = reinterpret_tensor(buf1198, (2048, 1024, 196), (200704, 196, 1), 0); del buf1198  # reuse
        buf1278 = buf1170; del buf1170  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_86.run(buf1269, buf1272, buf1278, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1272, buf1270, buf1271, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf1275 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1279 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87.run(buf1271, buf1270, buf1275, buf1279, 1024, stream=stream0)
        buf1276 = buf1272; del buf1272  # reuse
        buf1277 = reinterpret_tensor(buf1269, (2048, 1024, 196), (200704, 196, 1), 0); del buf1269  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1278, buf1279, buf1275, primals_220, primals_221, buf1276, buf1277, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_221
        buf1282 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1283 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1284 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1277, (802816, 512), (512, 1), 0), buf1282, buf1283, buf1284, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1288 = reinterpret_tensor(buf1277, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1277  # reuse
        buf1289 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf969, buf1289, 411041792, stream=stream0)
        buf1290 = reinterpret_tensor(buf1278, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1278  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_89.run(buf1276, buf1182, buf1290, 411041792, stream=stream0)
        del buf1182
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1290, buf1288, reinterpret_tensor(buf1289, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf1293 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_71.run(buf76, buf1293, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1289, (802816, 512), (512, 1), 0), reinterpret_tensor(buf1293, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf1295 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_224, buf1295, 262144, stream=stream0)
        del primals_224
        buf1297 = empty_strided_cuda((2048, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_90.run(buf1288, buf1297, 2097152, 16, stream=stream0)
        buf1298 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1299 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1300 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1288, (802816, 512), (512, 1), 0), buf1298, buf1299, buf1300, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1304 = reinterpret_tensor(buf1290, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1290  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_91.run(buf1288, buf1304, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten.convolution]
        buf1305 = extern_kernels.convolution(buf1304, buf1295, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1305, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1306 = buf1244; del buf1244  # reuse
        buf1307 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1341 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1342 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_25.run(buf125, buf1306, buf1307, buf1341, buf1342, 256, stream=stream0)
        buf1308 = reinterpret_tensor(buf1268, (2048, 256, 196), (50176, 196, 1), 0); del buf1268  # reuse
        buf1314 = reinterpret_tensor(buf1253, (2048, 256, 196), (50176, 196, 1), 0); del buf1253  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf1305, buf1308, buf1314, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1308, buf1306, buf1307, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf1311 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1315 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf1307, buf1306, buf1311, buf1315, 256, stream=stream0)
        buf1312 = buf1308; del buf1308  # reuse
        buf1313 = reinterpret_tensor(buf1305, (2048, 256, 196), (50176, 196, 1), 0); del buf1305  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1314, buf1315, buf1311, primals_226, primals_227, buf1312, buf1313, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf1314
        del primals_227
        buf1318 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1319 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1320 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1313, (200704, 512), (512, 1), 0), buf1318, buf1319, buf1320, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1324 = reinterpret_tensor(buf1313, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1313  # reuse
        buf1325 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf1360 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_92.run(buf904, buf1325, buf1360, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1312, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1324, reinterpret_tensor(buf1325, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf1328 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        buf1363 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_93.run(buf909, buf1328, buf1363, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1325, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1328, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1330 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_4_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(primals_230, buf1330, 65536, 9, stream=stream0)
        del primals_230
        buf1332 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_4_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1324, buf1332, 524288, 16, stream=stream0)
        buf1333 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1334 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1335 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1324, (200704, 512), (512, 1), 0), buf1333, buf1334, buf1335, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1339 = reinterpret_tensor(buf1312, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1312  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1324, buf1339, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_4_conv2], Original ATen: [aten.convolution]
        buf1340 = extern_kernels.convolution(buf1339, buf1330, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1340, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1343 = reinterpret_tensor(buf1339, (2048, 256, 196), (50176, 196, 1), 0); del buf1339  # reuse
        buf1349 = reinterpret_tensor(buf1324, (2048, 256, 196), (50176, 196, 1), 0); del buf1324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf1340, buf1343, buf1349, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1343, buf1341, buf1342, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf1346 = buf1315; del buf1315  # reuse
        buf1350 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf1342, buf1341, buf1346, buf1350, 256, stream=stream0)
        buf1347 = buf1343; del buf1343  # reuse
        buf1348 = reinterpret_tensor(buf1340, (2048, 256, 196), (50176, 196, 1), 0); del buf1340  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1349, buf1350, buf1346, primals_232, primals_233, buf1347, buf1348, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf1349
        del primals_233
        buf1353 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1354 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1355 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1348, (200704, 512), (512, 1), 0), buf1353, buf1354, buf1355, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1359 = reinterpret_tensor(buf1348, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1348  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1347, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1359, reinterpret_tensor(buf1360, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1360, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1363, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1365 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_4_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_236, buf1365, 262144, stream=stream0)
        del primals_236
        buf1367 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_4_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1359, buf1367, 524288, 16, stream=stream0)
        buf1368 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1369 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1370 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1359, (200704, 512), (512, 1), 0), buf1368, buf1369, buf1370, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1374 = reinterpret_tensor(buf1347, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1347  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1359, buf1374, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_4_conv3], Original ATen: [aten.convolution]
        buf1375 = extern_kernels.convolution(buf1374, buf1365, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1375, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf1376 = buf1279; del buf1279  # reuse
        buf1377 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_94.run(buf923, buf1376, buf1377, 1024, stream=stream0)
        buf1378 = reinterpret_tensor(buf1304, (2048, 1024, 196), (200704, 196, 1), 0); del buf1304  # reuse
        buf1384 = buf1276; del buf1276  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_86.run(buf1375, buf1378, buf1384, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1378, buf1376, buf1377, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf1381 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1385 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87.run(buf1377, buf1376, buf1381, buf1385, 1024, stream=stream0)
        buf1382 = buf1378; del buf1378  # reuse
        buf1383 = reinterpret_tensor(buf1375, (2048, 1024, 196), (200704, 196, 1), 0); del buf1375  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1384, buf1385, buf1381, primals_238, primals_239, buf1382, buf1383, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_239
        buf1388 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1389 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1390 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1383, (802816, 512), (512, 1), 0), buf1388, buf1389, buf1390, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1394 = reinterpret_tensor(buf1383, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1383  # reuse
        buf1395 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf969, buf1395, 411041792, stream=stream0)
        buf1396 = reinterpret_tensor(buf1384, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1384  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_89.run(buf1382, buf1288, buf1396, 411041792, stream=stream0)
        del buf1288
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1396, buf1394, reinterpret_tensor(buf1395, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), 411041792, 1024, 401408, 1, 1, stream=stream0)
        buf1399 = empty_strided_cuda((12845056, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_71.run(buf76, buf1399, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1395, (802816, 512), (512, 1), 0), reinterpret_tensor(buf1399, (802816, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        buf1401 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_242, buf1401, 262144, stream=stream0)
        del primals_242
        buf1403 = empty_strided_cuda((2048, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_90.run(buf1394, buf1403, 2097152, 16, stream=stream0)
        buf1404 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1405 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1406 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1394, (802816, 512), (512, 1), 0), buf1404, buf1405, buf1406, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1410 = reinterpret_tensor(buf1396, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1396  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_91.run(buf1394, buf1410, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten.convolution]
        buf1411 = extern_kernels.convolution(buf1410, buf1401, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1411, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1412 = buf1350; del buf1350  # reuse
        buf1413 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1447 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_95.run(buf125, buf1412, buf1413, buf1447, 256, stream=stream0)
        buf1414 = reinterpret_tensor(buf1374, (2048, 256, 196), (50176, 196, 1), 0); del buf1374  # reuse
        buf1420 = reinterpret_tensor(buf1359, (2048, 256, 196), (50176, 196, 1), 0); del buf1359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf1411, buf1414, buf1420, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1414, buf1412, buf1413, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf1417 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1421 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf1413, buf1412, buf1417, buf1421, 256, stream=stream0)
        buf1418 = buf1414; del buf1414  # reuse
        buf1419 = reinterpret_tensor(buf1411, (2048, 256, 196), (50176, 196, 1), 0); del buf1411  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1420, buf1421, buf1417, primals_244, primals_245, buf1418, buf1419, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf1420
        del primals_245
        buf1424 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1425 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1426 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1419, (200704, 512), (512, 1), 0), buf1424, buf1425, buf1426, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1430 = reinterpret_tensor(buf1419, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1419  # reuse
        buf1431 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf904, buf1431, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1418, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1430, reinterpret_tensor(buf1431, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), 102760448, 1024, 100352, 1, 1, stream=stream0)
        buf1434 = empty_strided_cuda((3211264, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_80.run(buf909, buf1434, 3211264, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1431, (200704, 512), (512, 1), 0), reinterpret_tensor(buf1434, (200704, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        buf1436 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_5_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(primals_248, buf1436, 65536, 9, stream=stream0)
        del primals_248
        buf1438 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_5_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1430, buf1438, 524288, 16, stream=stream0)
        buf1439 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1440 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1441 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1430, (200704, 512), (512, 1), 0), buf1439, buf1440, buf1441, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1445 = reinterpret_tensor(buf1418, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1418  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1430, buf1445, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer3_5_conv2], Original ATen: [aten.convolution]
        buf1446 = extern_kernels.convolution(buf1445, buf1436, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1446, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf1448 = reinterpret_tensor(buf1445, (2048, 256, 196), (50176, 196, 1), 0); del buf1445  # reuse
        buf1454 = reinterpret_tensor(buf1430, (2048, 256, 196), (50176, 196, 1), 0); del buf1430  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_75.run(buf1446, buf1448, buf1454, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_17.run(buf1448, buf125, buf1447, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf1451 = buf1421; del buf1421  # reuse
        buf1455 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_76.run(buf1447, buf125, buf1451, buf1455, 256, stream=stream0)
        buf1452 = buf1448; del buf1448  # reuse
        buf1453 = reinterpret_tensor(buf1446, (2048, 256, 196), (50176, 196, 1), 0); del buf1446  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_18.run(buf1454, buf1455, buf1451, primals_250, primals_251, buf1452, buf1453, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf1454
        del buf1455
        del primals_251
        buf1458 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1459 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1460 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1453, (200704, 512), (512, 1), 0), buf1458, buf1459, buf1460, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1464 = reinterpret_tensor(buf1453, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf1453  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1452, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf1464, buf904, 102760448, 1024, 100352, 1, 1, stream=stream0)
        del buf1007
        del buf1042
        del buf1113
        del buf1148
        del buf1219
        del buf1254
        del buf1325
        del buf1360
        del buf1431
        del buf906
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf904, (200704, 512), (512, 1), 0), buf909, 512, 1, 16, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del buf904
        buf1468 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_5_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(primals_254, buf1468, 262144, stream=stream0)
        del primals_254
        buf1470 = empty_strided_cuda((2048, 256, 4, 4), (4096, 1, 1024, 256), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer3_5_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_82.run(buf1464, buf1470, 524288, 16, stream=stream0)
        buf1471 = empty_strided_cuda((200704, 32), (32, 1), torch.int32)
        buf1472 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        buf1473 = empty_strided_cuda((200704, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1464, (200704, 512), (512, 1), 0), buf1471, buf1472, buf1473, 512, 1, 32, 1, 2, 16, 32, 3, 200704, 1, 1, stream=stream0)
        buf1477 = reinterpret_tensor(buf1452, (2048, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf1452  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_83.run(buf1464, buf1477, 524288, 196, stream=stream0)
        del buf1464
        # Topologically Sorted Source Nodes: [layer3_5_conv3], Original ATen: [aten.convolution]
        buf1478 = extern_kernels.convolution(buf1477, buf1468, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1478, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        del buf1477
        buf1479 = buf1385; del buf1385  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_96.run(buf923, buf1479, 1024, stream=stream0)
        buf1480 = reinterpret_tensor(buf1410, (2048, 1024, 196), (200704, 196, 1), 0); del buf1410  # reuse
        buf1486 = buf1382; del buf1382  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_86.run(buf1478, buf1480, buf1486, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_19.run(buf1480, buf923, buf1479, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf1483 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf1487 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_87.run(buf1479, buf923, buf1483, buf1487, 1024, stream=stream0)
        buf1484 = buf1480; del buf1480  # reuse
        buf1485 = reinterpret_tensor(buf1478, (2048, 1024, 196), (200704, 196, 1), 0); del buf1478  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_20.run(buf1486, buf1487, buf1483, primals_256, primals_257, buf1484, buf1485, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del buf1487
        del primals_257
        buf1490 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1491 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1492 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1485, (802816, 512), (512, 1), 0), buf1490, buf1491, buf1492, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1496 = reinterpret_tensor(buf1485, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1485  # reuse
        buf1497 = reinterpret_tensor(buf1486, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf1486  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_89.run(buf1484, buf1394, buf1497, 411041792, stream=stream0)
        del buf1394
        del buf1484
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1497, buf1496, buf969, 411041792, 1024, 401408, 1, 1, stream=stream0)
        del buf1077
        del buf1183
        del buf1289
        del buf1395
        del buf971
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf969, (802816, 512), (512, 1), 0), buf76, 512, 1, 16, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del buf969
        buf1501 = empty_strided_cuda((512, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_88.run(primals_260, buf1501, 524288, stream=stream0)
        del primals_260
        buf1503 = empty_strided_cuda((2048, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_90.run(buf1496, buf1503, 2097152, 16, stream=stream0)
        buf1504 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1505 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1506 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1496, (802816, 512), (512, 1), 0), buf1504, buf1505, buf1506, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1510 = reinterpret_tensor(buf1497, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf1497  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_91.run(buf1496, buf1510, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten.convolution]
        buf1511 = extern_kernels.convolution(buf1510, buf1501, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1511, (2048, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution.default')
        buf1512 = buf826; del buf826  # reuse
        buf1513 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1547 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1548 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf472, buf1512, buf1513, buf1547, buf1548, 512, stream=stream0)
        buf1514 = reinterpret_tensor(buf815, (2048, 512, 196), (100352, 196, 1), 0); del buf815  # reuse
        buf1520 = reinterpret_tensor(buf801, (2048, 512, 196), (100352, 196, 1), 0); del buf801  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_97.run(buf1511, buf1514, buf1520, 1048576, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_21.run(buf1514, buf1512, buf1513, 401408, 196, 100352, 196, 1024, 512, 392, 1, stream=stream0)
        buf1517 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1521 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer4_0_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_98.run(buf1513, buf1512, buf1517, buf1521, 512, stream=stream0)
        buf1518 = buf1514; del buf1514  # reuse
        buf1519 = reinterpret_tensor(buf1511, (2048, 512, 196), (100352, 196, 1), 0); del buf1511  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_22.run(buf1520, buf1521, buf1517, primals_262, primals_263, buf1518, buf1519, 401408, 196, 100352, 196, 1024, 512, 392, 1, stream=stream0)
        del buf1520
        del primals_263
        buf1524 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf1525 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf1526 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1519, (401408, 512), (512, 1), 0), buf1524, buf1525, buf1526, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf1530 = reinterpret_tensor(buf1519, (2048, 512, 14, 14), (100352, 196, 14, 1), 0); del buf1519  # reuse
        buf1531 = reinterpret_tensor(buf453, (2048, 512, 14, 14), (100352, 196, 14, 1), 0); del buf453  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_49.run(buf1531, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1518, (2048, 512, 14, 14), (100352, 196, 14, 1), 0), buf1530, buf1531, 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf1534 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_52.run(buf458, buf1534, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1531, (401408, 512), (512, 1), 0), reinterpret_tensor(buf1534, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf1536 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_99.run(primals_266, buf1536, 262144, 9, stream=stream0)
        del primals_266
        buf1538 = empty_strided_cuda((2048, 512, 4, 4), (8192, 1, 2048, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_100.run(buf1530, buf1538, 1048576, 16, stream=stream0)
        buf1539 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf1540 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf1541 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1530, (401408, 512), (512, 1), 0), buf1539, buf1540, buf1541, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf1545 = reinterpret_tensor(buf1518, (2048, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf1518  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_101.run(buf1530, buf1545, 1048576, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_0_conv2], Original ATen: [aten.convolution]
        buf1546 = extern_kernels.convolution(buf1545, buf1536, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1546, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1549 = empty_strided_cuda((2048, 512, 49), (25088, 49, 1), torch.bfloat16)
        buf1555 = empty_strided_cuda((2048, 512, 49), (25088, 49, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_102.run(buf1546, buf1549, buf1555, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1549, buf1547, buf1548, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf1552 = buf1521; del buf1521  # reuse
        buf1556 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103.run(buf1548, buf1547, buf1552, buf1556, 512, stream=stream0)
        buf1553 = buf1549; del buf1549  # reuse
        buf1554 = reinterpret_tensor(buf1546, (2048, 512, 49), (25088, 49, 1), 0); del buf1546  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1555, buf1556, buf1552, primals_268, primals_269, buf1553, buf1554, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del buf1555
        del primals_269
        buf1559 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1560 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1561 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1554, (100352, 512), (512, 1), 0), buf1559, buf1560, buf1561, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1565 = empty_strided_cuda((2048, 512, 7, 7), (25088, 49, 7, 1), torch.int8)
        # Topologically Sorted Source Nodes: [layer4_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_104.run(buf1565, 51380224, stream=stream0)
        buf1566 = reinterpret_tensor(buf1554, (2048, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1554  # reuse
        buf1567 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_105.run(buf1565, buf1567, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1553, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf1566, reinterpret_tensor(buf1567, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf1570 = empty_strided_cuda((100352, 16), (16, 1), torch.int32)
        # Topologically Sorted Source Nodes: [layer4_0_relu_1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_106.run(buf1570, 1605632, stream=stream0)
        buf1571 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_107.run(buf1570, buf1571, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1567, (100352, 512), (512, 1), 0), reinterpret_tensor(buf1571, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf1573 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_108.run(primals_272, buf1573, 1048576, stream=stream0)
        del primals_272
        buf1575 = empty_strided_cuda((2048, 512, 2, 2), (2048, 1, 1024, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_109.run(buf1566, buf1575, 1048576, 4, stream=stream0)
        buf1576 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1577 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1578 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1566, (100352, 512), (512, 1), 0), buf1576, buf1577, buf1578, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1582 = reinterpret_tensor(buf1553, (2048, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1553  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_110.run(buf1566, buf1582, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_0_conv3], Original ATen: [aten.convolution]
        buf1583 = extern_kernels.convolution(buf1582, buf1573, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1583, (2048, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        buf1584 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_111.run(buf1584, 2048, stream=stream0)
        buf1585 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1586 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1612 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1613 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_112.run(buf1584, buf1585, buf1586, buf1612, buf1613, 2048, stream=stream0)
        buf1587 = reinterpret_tensor(buf1545, (2048, 2048, 49), (100352, 49, 1), 0); del buf1545  # reuse
        buf1593 = reinterpret_tensor(buf1530, (2048, 2048, 49), (100352, 49, 1), 0); del buf1530  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_113.run(buf1583, buf1587, buf1593, 4194304, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_25.run(buf1587, buf1585, buf1586, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        buf1590 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1594 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_114.run(buf1586, buf1585, buf1590, buf1594, 2048, stream=stream0)
        buf1591 = buf1587; del buf1587  # reuse
        buf1592 = reinterpret_tensor(buf1583, (2048, 2048, 49), (100352, 49, 1), 0); del buf1583  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_26.run(buf1593, buf1594, buf1590, primals_274, primals_275, buf1591, buf1592, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        del primals_275
        buf1597 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf1598 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf1599 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1592, (401408, 512), (512, 1), 0), buf1597, buf1598, buf1599, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf1603 = empty_strided_cuda((2048, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_115.run(primals_278, buf1603, 2097152, stream=stream0)
        del primals_278
        buf1604 = empty_strided_cuda((802816, 32), (32, 1), torch.int32)
        buf1605 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        buf1606 = empty_strided_cuda((802816, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1496, (802816, 512), (512, 1), 0), buf1604, buf1605, buf1606, 512, 1, 32, 1, 2, 16, 32, 3, 802816, 1, 1, stream=stream0)
        buf1610 = buf1510; del buf1510  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_91.run(buf1496, buf1610, 2097152, 196, stream=stream0)
        del buf1496
        # Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten.convolution]
        buf1611 = extern_kernels.convolution(buf1610, buf1603, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1611, (2048, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        del buf1610
        buf1614 = buf1592; del buf1592  # reuse
        buf1620 = buf1593; del buf1593  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_113.run(buf1611, buf1614, buf1620, 4194304, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_25.run(buf1614, buf1612, buf1613, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        buf1617 = buf1594; del buf1594  # reuse
        buf1621 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_downsample_1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_114.run(buf1613, buf1612, buf1617, buf1621, 2048, stream=stream0)
        buf1618 = buf1614; del buf1614  # reuse
        buf1619 = reinterpret_tensor(buf1611, (2048, 2048, 49), (100352, 49, 1), 0); del buf1611  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_26.run(buf1620, buf1621, buf1617, primals_280, primals_281, buf1618, buf1619, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        del primals_281
        buf1624 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf1625 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf1626 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1619, (401408, 512), (512, 1), 0), buf1624, buf1625, buf1626, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf1630 = reinterpret_tensor(buf1531, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1531  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_relu_2], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_49.run(buf1630, 205520896, stream=stream0)
        buf1631 = reinterpret_tensor(buf1619, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1619  # reuse
        buf1632 = buf768; del buf768  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_50.run(buf1630, buf1632, 205520896, stream=stream0)
        buf1633 = reinterpret_tensor(buf1620, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1620  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_116.run(buf1591, buf1618, buf1633, 205520896, stream=stream0)
        del buf1591
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1633, buf1631, reinterpret_tensor(buf1632, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf1636 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_52.run(buf458, buf1636, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1632, (401408, 512), (512, 1), 0), reinterpret_tensor(buf1636, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf1638 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_108.run(primals_284, buf1638, 1048576, stream=stream0)
        del primals_284
        buf1640 = empty_strided_cuda((2048, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_117.run(buf1631, buf1640, 4194304, 4, stream=stream0)
        buf1641 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf1642 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf1643 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1631, (401408, 512), (512, 1), 0), buf1641, buf1642, buf1643, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf1647 = reinterpret_tensor(buf1633, (2048, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1633  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_118.run(buf1631, buf1647, 4194304, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten.convolution]
        buf1648 = extern_kernels.convolution(buf1647, buf1638, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1648, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1649 = buf1556; del buf1556  # reuse
        buf1650 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1684 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1685 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf472, buf1649, buf1650, buf1684, buf1685, 512, stream=stream0)
        buf1651 = reinterpret_tensor(buf1582, (2048, 512, 49), (25088, 49, 1), 0); del buf1582  # reuse
        buf1657 = reinterpret_tensor(buf1566, (2048, 512, 49), (25088, 49, 1), 0); del buf1566  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_102.run(buf1648, buf1651, buf1657, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1651, buf1649, buf1650, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf1654 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1658 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103.run(buf1650, buf1649, buf1654, buf1658, 512, stream=stream0)
        buf1655 = buf1651; del buf1651  # reuse
        buf1656 = reinterpret_tensor(buf1648, (2048, 512, 49), (25088, 49, 1), 0); del buf1648  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1657, buf1658, buf1654, primals_286, primals_287, buf1655, buf1656, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del buf1657
        del primals_287
        buf1661 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1662 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1663 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1656, (100352, 512), (512, 1), 0), buf1661, buf1662, buf1663, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1667 = reinterpret_tensor(buf1656, (2048, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1656  # reuse
        buf1668 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf1703 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_119.run(buf1565, buf1668, buf1703, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1655, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf1667, reinterpret_tensor(buf1668, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf1671 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        buf1706 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_120.run(buf1570, buf1671, buf1706, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1668, (100352, 512), (512, 1), 0), reinterpret_tensor(buf1671, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf1673 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_1_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_99.run(primals_290, buf1673, 262144, 9, stream=stream0)
        del primals_290
        buf1675 = empty_strided_cuda((2048, 512, 2, 2), (2048, 1, 1024, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer4_1_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_109.run(buf1667, buf1675, 1048576, 4, stream=stream0)
        buf1676 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1677 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1678 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1667, (100352, 512), (512, 1), 0), buf1676, buf1677, buf1678, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1682 = reinterpret_tensor(buf1655, (2048, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1655  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_110.run(buf1667, buf1682, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_1_conv2], Original ATen: [aten.convolution]
        buf1683 = extern_kernels.convolution(buf1682, buf1673, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1683, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1686 = reinterpret_tensor(buf1682, (2048, 512, 49), (25088, 49, 1), 0); del buf1682  # reuse
        buf1692 = reinterpret_tensor(buf1667, (2048, 512, 49), (25088, 49, 1), 0); del buf1667  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_102.run(buf1683, buf1686, buf1692, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1686, buf1684, buf1685, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf1689 = buf1658; del buf1658  # reuse
        buf1693 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103.run(buf1685, buf1684, buf1689, buf1693, 512, stream=stream0)
        buf1690 = buf1686; del buf1686  # reuse
        buf1691 = reinterpret_tensor(buf1683, (2048, 512, 49), (25088, 49, 1), 0); del buf1683  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1692, buf1693, buf1689, primals_292, primals_293, buf1690, buf1691, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del buf1692
        del primals_293
        buf1696 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1697 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1698 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1691, (100352, 512), (512, 1), 0), buf1696, buf1697, buf1698, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1702 = reinterpret_tensor(buf1691, (2048, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1691  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1690, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf1702, reinterpret_tensor(buf1703, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1703, (100352, 512), (512, 1), 0), reinterpret_tensor(buf1706, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf1708 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_1_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_108.run(primals_296, buf1708, 1048576, stream=stream0)
        del primals_296
        buf1710 = empty_strided_cuda((2048, 512, 2, 2), (2048, 1, 1024, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer4_1_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_109.run(buf1702, buf1710, 1048576, 4, stream=stream0)
        buf1711 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1712 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1713 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1702, (100352, 512), (512, 1), 0), buf1711, buf1712, buf1713, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1717 = reinterpret_tensor(buf1690, (2048, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1690  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_110.run(buf1702, buf1717, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_1_conv3], Original ATen: [aten.convolution]
        buf1718 = extern_kernels.convolution(buf1717, buf1708, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1718, (2048, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        buf1719 = buf1621; del buf1621  # reuse
        buf1720 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_121.run(buf1584, buf1719, buf1720, 2048, stream=stream0)
        buf1721 = reinterpret_tensor(buf1647, (2048, 2048, 49), (100352, 49, 1), 0); del buf1647  # reuse
        buf1727 = buf1618; del buf1618  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_113.run(buf1718, buf1721, buf1727, 4194304, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_25.run(buf1721, buf1719, buf1720, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        buf1724 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1728 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_114.run(buf1720, buf1719, buf1724, buf1728, 2048, stream=stream0)
        buf1725 = buf1721; del buf1721  # reuse
        buf1726 = reinterpret_tensor(buf1718, (2048, 2048, 49), (100352, 49, 1), 0); del buf1718  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_26.run(buf1727, buf1728, buf1724, primals_298, primals_299, buf1725, buf1726, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        del primals_299
        buf1731 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf1732 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf1733 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1726, (401408, 512), (512, 1), 0), buf1731, buf1732, buf1733, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf1737 = reinterpret_tensor(buf1726, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1726  # reuse
        buf1738 = buf697; del buf697  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_50.run(buf1630, buf1738, 205520896, stream=stream0)
        buf1739 = reinterpret_tensor(buf1727, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1727  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_116.run(buf1725, buf1631, buf1739, 205520896, stream=stream0)
        del buf1631
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1739, buf1737, reinterpret_tensor(buf1738, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), 205520896, 1024, 200704, 1, 1, stream=stream0)
        buf1742 = empty_strided_cuda((6422528, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_52.run(buf458, buf1742, 6422528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1738, (401408, 512), (512, 1), 0), reinterpret_tensor(buf1742, (401408, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        buf1744 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_108.run(primals_302, buf1744, 1048576, stream=stream0)
        del primals_302
        buf1746 = empty_strided_cuda((2048, 2048, 2, 2), (8192, 1, 4096, 2048), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_117.run(buf1737, buf1746, 4194304, 4, stream=stream0)
        buf1747 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf1748 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf1749 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1737, (401408, 512), (512, 1), 0), buf1747, buf1748, buf1749, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf1753 = reinterpret_tensor(buf1739, (2048, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf1739  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_118.run(buf1737, buf1753, 4194304, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten.convolution]
        buf1754 = extern_kernels.convolution(buf1753, buf1744, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1754, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1755 = buf1693; del buf1693  # reuse
        buf1756 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1790 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_122.run(buf472, buf1755, buf1756, buf1790, 512, stream=stream0)
        buf1757 = reinterpret_tensor(buf1717, (2048, 512, 49), (25088, 49, 1), 0); del buf1717  # reuse
        buf1763 = reinterpret_tensor(buf1702, (2048, 512, 49), (25088, 49, 1), 0); del buf1702  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_102.run(buf1754, buf1757, buf1763, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1757, buf1755, buf1756, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf1760 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf1764 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn1, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103.run(buf1756, buf1755, buf1760, buf1764, 512, stream=stream0)
        buf1761 = buf1757; del buf1757  # reuse
        buf1762 = reinterpret_tensor(buf1754, (2048, 512, 49), (25088, 49, 1), 0); del buf1754  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1763, buf1764, buf1760, primals_304, primals_305, buf1761, buf1762, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del buf1763
        del primals_305
        buf1767 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1768 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1769 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1762, (100352, 512), (512, 1), 0), buf1767, buf1768, buf1769, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1773 = reinterpret_tensor(buf1762, (2048, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1762  # reuse
        buf1774 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_105.run(buf1565, buf1774, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1761, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf1773, reinterpret_tensor(buf1774, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), 51380224, 1024, 50176, 1, 1, stream=stream0)
        buf1777 = empty_strided_cuda((1605632, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_107.run(buf1570, buf1777, 1605632, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1774, (100352, 512), (512, 1), 0), reinterpret_tensor(buf1777, (100352, 16), (16, 1), 0), 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        buf1779 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_2_conv2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_99.run(primals_308, buf1779, 262144, 9, stream=stream0)
        del primals_308
        buf1781 = empty_strided_cuda((2048, 512, 2, 2), (2048, 1, 1024, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer4_2_conv2], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_109.run(buf1773, buf1781, 1048576, 4, stream=stream0)
        buf1782 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1783 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1784 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1773, (100352, 512), (512, 1), 0), buf1782, buf1783, buf1784, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1788 = reinterpret_tensor(buf1761, (2048, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1761  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_conv2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_110.run(buf1773, buf1788, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [layer4_2_conv2], Original ATen: [aten.convolution]
        buf1789 = extern_kernels.convolution(buf1788, buf1779, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1789, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf1791 = reinterpret_tensor(buf1788, (2048, 512, 49), (25088, 49, 1), 0); del buf1788  # reuse
        buf1797 = reinterpret_tensor(buf1773, (2048, 512, 49), (25088, 49, 1), 0); del buf1773  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_102.run(buf1789, buf1791, buf1797, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_23.run(buf1791, buf472, buf1790, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf1794 = buf1764; del buf1764  # reuse
        buf1798 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn2, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_103.run(buf1790, buf472, buf1794, buf1798, 512, stream=stream0)
        buf1795 = buf1791; del buf1791  # reuse
        buf1796 = reinterpret_tensor(buf1789, (2048, 512, 49), (25088, 49, 1), 0); del buf1789  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_24.run(buf1797, buf1798, buf1794, primals_310, primals_311, buf1795, buf1796, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del buf1797
        del buf1798
        del primals_311
        buf1801 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1802 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1803 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1796, (100352, 512), (512, 1), 0), buf1801, buf1802, buf1803, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1807 = reinterpret_tensor(buf1796, (2048, 512, 7, 7), (25088, 49, 7, 1), 0); del buf1796  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(reinterpret_tensor(buf1795, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf1807, buf1565, 51380224, 1024, 50176, 1, 1, stream=stream0)
        del buf1567
        del buf1668
        del buf1703
        del buf1774
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1565, (100352, 512), (512, 1), 0), buf1570, 512, 1, 16, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del buf1565
        buf1811 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_2_conv3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_108.run(primals_314, buf1811, 1048576, stream=stream0)
        del primals_314
        buf1813 = empty_strided_cuda((2048, 512, 2, 2), (2048, 1, 1024, 512), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [layer4_2_conv3], Original ATen: [aten.avg_pool2d, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_avg_pool2d_109.run(buf1807, buf1813, 1048576, 4, stream=stream0)
        buf1814 = empty_strided_cuda((100352, 32), (32, 1), torch.int32)
        buf1815 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        buf1816 = empty_strided_cuda((100352, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1807, (100352, 512), (512, 1), 0), buf1814, buf1815, buf1816, 512, 1, 32, 1, 2, 16, 32, 3, 100352, 1, 1, stream=stream0)
        buf1820 = reinterpret_tensor(buf1795, (2048, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf1795  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_conv3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_110.run(buf1807, buf1820, 1048576, 49, stream=stream0)
        del buf1807
        # Topologically Sorted Source Nodes: [layer4_2_conv3], Original ATen: [aten.convolution]
        buf1821 = extern_kernels.convolution(buf1820, buf1811, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1821, (2048, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        del buf1820
        buf1822 = buf1728; del buf1728  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_123.run(buf1584, buf1822, 2048, stream=stream0)
        buf1823 = reinterpret_tensor(buf1753, (2048, 2048, 49), (100352, 49, 1), 0); del buf1753  # reuse
        buf1829 = buf1725; del buf1725  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_113.run(buf1821, buf1823, buf1829, 4194304, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_reduce_kernel_25.run(buf1823, buf1584, buf1822, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        buf1826 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf1830 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn3, ], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsqrt_sub_114.run(buf1822, buf1584, buf1826, buf1830, 2048, stream=stream0)
        buf1827 = buf1823; del buf1823  # reuse
        buf1828 = reinterpret_tensor(buf1821, (2048, 2048, 49), (100352, 49, 1), 0); del buf1821  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_fwd_norm_kernel_26.run(buf1829, buf1830, buf1826, primals_316, primals_317, buf1827, buf1828, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        del buf1830
        del primals_317
        buf1833 = empty_strided_cuda((401408, 32), (32, 1), torch.int32)
        buf1834 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        buf1835 = empty_strided_cuda((401408, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1828, (401408, 512), (512, 1), 0), buf1833, buf1834, buf1835, 512, 1, 32, 1, 2, 16, 32, 3, 401408, 1, 1, stream=stream0)
        buf1839 = reinterpret_tensor(buf1828, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1828  # reuse
        buf1840 = reinterpret_tensor(buf1829, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf1829  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_116.run(buf1827, buf1737, buf1840, 205520896, stream=stream0)
        del buf1737
        del buf1827
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        relu_kernel_3.run(buf1840, buf1839, buf1630, 205520896, 1024, 200704, 1, 1, stream=stream0)
        del buf1632
        del buf1738
        del buf1840
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        pack_kernel_4.run(reinterpret_tensor(buf1630, (401408, 512), (512, 1), 0), buf458, 512, 1, 16, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del buf1630
        buf1845 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [avgpool, flatten], Original ATen: [aten.mean, aten.view]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_view_124.run(buf1839, buf1845, 4194304, 49, stream=stream0)
        del buf1839
        buf1846 = empty_strided_cuda((8192, 32), (32, 1), torch.int32)
        buf1847 = empty_strided_cuda((8192, ), (1, ), torch.bfloat16)
        buf1848 = empty_strided_cuda((8192, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        quant_pack_kernel_0.run(reinterpret_tensor(buf1845, (8192, 512), (512, 1), 0), buf1846, buf1847, buf1848, 512, 1, 32, 1, 2, 16, 32, 3, 8192, 1, 1, stream=stream0)
        buf1852 = empty_strided_cuda((100, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [fc], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_125.run(primals_320, buf1852, 204800, stream=stream0)
        del primals_320
        buf1853 = empty_strided_cuda((2048, 100), (100, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [fc], Original ATen: [aten.mm]
        extern_kernels.mm(buf1845, reinterpret_tensor(buf1852, (2048, 100), (1, 2048), 0), out=buf1853)
        del buf1845
        # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_3, primals_3, 1, stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__div_mul_sub_127.run(primals_6, buf13, primals_7, buf14, primals_6, primals_7, 64, stream=stream0)
        del buf13
        del buf14
        del primals_6
        del primals_7
        # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_9, primals_9, 1, stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [layer1_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_128.run(primals_12, buf53, primals_13, buf54, primals_12, primals_13, 64, stream=stream0)
        del buf53
        del buf54
        del primals_12
        del primals_13
        # Topologically Sorted Source Nodes: [add__2], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_15, primals_15, 1, stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_128.run(primals_18, buf90, primals_19, buf91, primals_18, primals_19, 64, stream=stream0)
        del buf90
        del buf91
        del primals_18
        del primals_19
        # Topologically Sorted Source Nodes: [add__3], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_21, primals_21, 1, stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_129.run(primals_24, buf126, primals_25, buf127, primals_24, primals_25, 256, stream=stream0)
        del buf126
        del buf127
        del primals_24
        del primals_25
        # Topologically Sorted Source Nodes: [add__4], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_27, primals_27, 1, stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_downsample_1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_129.run(primals_30, buf153, primals_31, buf154, primals_30, primals_31, 256, stream=stream0)
        del buf153
        del buf154
        del primals_30
        del primals_31
        # Topologically Sorted Source Nodes: [add__5], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_33, primals_33, 1, stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_128.run(primals_36, buf190, primals_37, buf191, primals_36, primals_37, 64, stream=stream0)
        del buf190
        del buf191
        del primals_36
        del primals_37
        # Topologically Sorted Source Nodes: [add__6], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_39, primals_39, 1, stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_128.run(primals_42, buf225, primals_43, buf226, primals_42, primals_43, 64, stream=stream0)
        del buf225
        del buf226
        del primals_42
        del primals_43
        # Topologically Sorted Source Nodes: [add__7], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_45, primals_45, 1, stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_1_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_129.run(primals_48, buf260, primals_49, buf261, primals_48, primals_49, 256, stream=stream0)
        del buf260
        del buf261
        del primals_48
        del primals_49
        # Topologically Sorted Source Nodes: [add__8], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_51, primals_51, 1, stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_128.run(primals_54, buf296, primals_55, buf297, primals_54, primals_55, 64, stream=stream0)
        del buf296
        del buf297
        del primals_54
        del primals_55
        # Topologically Sorted Source Nodes: [add__9], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_57, primals_57, 1, stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_128.run(primals_60, buf12, primals_61, buf331, primals_60, primals_61, 64, stream=stream0)
        del buf12
        del buf331
        del primals_60
        del primals_61
        # Topologically Sorted Source Nodes: [add__10], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_63, primals_63, 1, stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_2_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_129.run(primals_66, buf364, primals_67, buf365, primals_66, primals_67, 256, stream=stream0)
        del buf364
        del buf365
        del primals_66
        del primals_67
        # Topologically Sorted Source Nodes: [add__11], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_69, primals_69, 1, stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer2_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_130.run(primals_72, buf399, primals_73, buf400, primals_72, primals_73, 128, stream=stream0)
        del buf399
        del buf400
        del primals_72
        del primals_73
        # Topologically Sorted Source Nodes: [add__12], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_75, primals_75, 1, stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [layer2_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_131.run(primals_78, buf435, primals_79, buf436, primals_78, primals_79, 128, stream=stream0)
        del buf435
        del buf436
        del primals_78
        del primals_79
        # Topologically Sorted Source Nodes: [add__13], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_81, primals_81, 1, stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_132.run(primals_84, buf473, primals_85, buf474, primals_84, primals_85, 512, stream=stream0)
        del buf473
        del buf474
        del primals_84
        del primals_85
        # Topologically Sorted Source Nodes: [add__14], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_87, primals_87, 1, stream=stream0)
        del primals_87
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_downsample_1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_132.run(primals_90, buf500, primals_91, buf501, primals_90, primals_91, 512, stream=stream0)
        del buf500
        del buf501
        del primals_90
        del primals_91
        # Topologically Sorted Source Nodes: [add__15], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_93, primals_93, 1, stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_131.run(primals_96, buf537, primals_97, buf538, primals_96, primals_97, 128, stream=stream0)
        del buf537
        del buf538
        del primals_96
        del primals_97
        # Topologically Sorted Source Nodes: [add__16], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_99, primals_99, 1, stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_131.run(primals_102, buf572, primals_103, buf573, primals_102, primals_103, 128, stream=stream0)
        del buf572
        del buf573
        del primals_102
        del primals_103
        # Topologically Sorted Source Nodes: [add__17], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_105, primals_105, 1, stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_1_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_132.run(primals_108, buf607, primals_109, buf608, primals_108, primals_109, 512, stream=stream0)
        del buf607
        del buf608
        del primals_108
        del primals_109
        # Topologically Sorted Source Nodes: [add__18], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_111, primals_111, 1, stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_131.run(primals_114, buf643, primals_115, buf644, primals_114, primals_115, 128, stream=stream0)
        del buf643
        del buf644
        del primals_114
        del primals_115
        # Topologically Sorted Source Nodes: [add__19], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_117, primals_117, 1, stream=stream0)
        del primals_117
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_131.run(primals_120, buf678, primals_121, buf679, primals_120, primals_121, 128, stream=stream0)
        del buf678
        del buf679
        del primals_120
        del primals_121
        # Topologically Sorted Source Nodes: [add__20], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_123, primals_123, 1, stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_2_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_132.run(primals_126, buf713, primals_127, buf714, primals_126, primals_127, 512, stream=stream0)
        del buf713
        del buf714
        del primals_126
        del primals_127
        # Topologically Sorted Source Nodes: [add__21], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_129, primals_129, 1, stream=stream0)
        del primals_129
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_131.run(primals_132, buf749, primals_133, buf750, primals_132, primals_133, 128, stream=stream0)
        del buf749
        del buf750
        del primals_132
        del primals_133
        # Topologically Sorted Source Nodes: [add__22], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_135, primals_135, 1, stream=stream0)
        del primals_135
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_131.run(primals_138, buf398, primals_139, buf784, primals_138, primals_139, 128, stream=stream0)
        del buf398
        del buf784
        del primals_138
        del primals_139
        # Topologically Sorted Source Nodes: [add__23], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_141, primals_141, 1, stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_3_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_132.run(primals_144, buf817, primals_145, buf818, primals_144, primals_145, 512, stream=stream0)
        del buf817
        del buf818
        del primals_144
        del primals_145
        # Topologically Sorted Source Nodes: [add__24], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_147, primals_147, 1, stream=stream0)
        del primals_147
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer3_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_133.run(primals_150, buf851, primals_151, buf852, primals_150, primals_151, 256, stream=stream0)
        del buf851
        del buf852
        del primals_150
        del primals_151
        # Topologically Sorted Source Nodes: [add__25], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_153, primals_153, 1, stream=stream0)
        del primals_153
        # Topologically Sorted Source Nodes: [layer3_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_156, buf886, primals_157, buf887, primals_156, primals_157, 256, stream=stream0)
        del buf886
        del buf887
        del primals_156
        del primals_157
        # Topologically Sorted Source Nodes: [add__26], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_159, primals_159, 1, stream=stream0)
        del primals_159
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_135.run(primals_162, buf924, primals_163, buf925, primals_162, primals_163, 1024, stream=stream0)
        del buf924
        del buf925
        del primals_162
        del primals_163
        # Topologically Sorted Source Nodes: [add__27], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_165, primals_165, 1, stream=stream0)
        del primals_165
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_downsample_1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_135.run(primals_168, buf951, primals_169, buf952, primals_168, primals_169, 1024, stream=stream0)
        del buf951
        del buf952
        del primals_168
        del primals_169
        # Topologically Sorted Source Nodes: [add__28], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_171, primals_171, 1, stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_174, buf988, primals_175, buf989, primals_174, primals_175, 256, stream=stream0)
        del buf988
        del buf989
        del primals_174
        del primals_175
        # Topologically Sorted Source Nodes: [add__29], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_177, primals_177, 1, stream=stream0)
        del primals_177
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_180, buf1023, primals_181, buf1024, primals_180, primals_181, 256, stream=stream0)
        del buf1023
        del buf1024
        del primals_180
        del primals_181
        # Topologically Sorted Source Nodes: [add__30], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_183, primals_183, 1, stream=stream0)
        del primals_183
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_1_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_135.run(primals_186, buf1058, primals_187, buf1059, primals_186, primals_187, 1024, stream=stream0)
        del buf1058
        del buf1059
        del primals_186
        del primals_187
        # Topologically Sorted Source Nodes: [add__31], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_189, primals_189, 1, stream=stream0)
        del primals_189
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_192, buf1094, primals_193, buf1095, primals_192, primals_193, 256, stream=stream0)
        del buf1094
        del buf1095
        del primals_192
        del primals_193
        # Topologically Sorted Source Nodes: [add__32], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_195, primals_195, 1, stream=stream0)
        del primals_195
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_198, buf1129, primals_199, buf1130, primals_198, primals_199, 256, stream=stream0)
        del buf1129
        del buf1130
        del primals_198
        del primals_199
        # Topologically Sorted Source Nodes: [add__33], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_201, primals_201, 1, stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_2_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_135.run(primals_204, buf1164, primals_205, buf1165, primals_204, primals_205, 1024, stream=stream0)
        del buf1164
        del buf1165
        del primals_204
        del primals_205
        # Topologically Sorted Source Nodes: [add__34], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_207, primals_207, 1, stream=stream0)
        del primals_207
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_210, buf1200, primals_211, buf1201, primals_210, primals_211, 256, stream=stream0)
        del buf1200
        del buf1201
        del primals_210
        del primals_211
        # Topologically Sorted Source Nodes: [add__35], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_213, primals_213, 1, stream=stream0)
        del primals_213
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_216, buf1235, primals_217, buf1236, primals_216, primals_217, 256, stream=stream0)
        del buf1235
        del buf1236
        del primals_216
        del primals_217
        # Topologically Sorted Source Nodes: [add__36], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_219, primals_219, 1, stream=stream0)
        del primals_219
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_3_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_135.run(primals_222, buf1270, primals_223, buf1271, primals_222, primals_223, 1024, stream=stream0)
        del buf1270
        del buf1271
        del primals_222
        del primals_223
        # Topologically Sorted Source Nodes: [add__37], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_225, primals_225, 1, stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_228, buf1306, primals_229, buf1307, primals_228, primals_229, 256, stream=stream0)
        del buf1306
        del buf1307
        del primals_228
        del primals_229
        # Topologically Sorted Source Nodes: [add__38], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_231, primals_231, 1, stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_234, buf1341, primals_235, buf1342, primals_234, primals_235, 256, stream=stream0)
        del buf1341
        del buf1342
        del primals_234
        del primals_235
        # Topologically Sorted Source Nodes: [add__39], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_237, primals_237, 1, stream=stream0)
        del primals_237
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_4_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_135.run(primals_240, buf1376, primals_241, buf1377, primals_240, primals_241, 1024, stream=stream0)
        del buf1376
        del buf1377
        del primals_240
        del primals_241
        # Topologically Sorted Source Nodes: [add__40], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_243, primals_243, 1, stream=stream0)
        del primals_243
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_246, buf1412, primals_247, buf1413, primals_246, primals_247, 256, stream=stream0)
        del buf1412
        del buf1413
        del primals_246
        del primals_247
        # Topologically Sorted Source Nodes: [add__41], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_249, primals_249, 1, stream=stream0)
        del primals_249
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_134.run(primals_252, buf125, primals_253, buf1447, primals_252, primals_253, 256, stream=stream0)
        del buf125
        del buf1447
        del primals_252
        del primals_253
        # Topologically Sorted Source Nodes: [add__42], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_255, primals_255, 1, stream=stream0)
        del primals_255
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_5_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_135.run(primals_258, buf923, primals_259, buf1479, primals_258, primals_259, 1024, stream=stream0)
        del buf1479
        del buf923
        del primals_258
        del primals_259
        # Topologically Sorted Source Nodes: [add__43], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_261, primals_261, 1, stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer4_0_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_136.run(primals_264, buf1512, primals_265, buf1513, primals_264, primals_265, 512, stream=stream0)
        del buf1512
        del buf1513
        del primals_264
        del primals_265
        # Topologically Sorted Source Nodes: [add__44], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_267, primals_267, 1, stream=stream0)
        del primals_267
        # Topologically Sorted Source Nodes: [layer4_0_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_137.run(primals_270, buf1547, primals_271, buf1548, primals_270, primals_271, 512, stream=stream0)
        del buf1547
        del buf1548
        del primals_270
        del primals_271
        # Topologically Sorted Source Nodes: [add__45], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_273, primals_273, 1, stream=stream0)
        del primals_273
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_138.run(primals_276, buf1585, primals_277, buf1586, primals_276, primals_277, 2048, stream=stream0)
        del buf1585
        del buf1586
        del primals_276
        del primals_277
        # Topologically Sorted Source Nodes: [add__46], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_279, primals_279, 1, stream=stream0)
        del primals_279
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_downsample_1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_138.run(primals_282, buf1612, primals_283, buf1613, primals_282, primals_283, 2048, stream=stream0)
        del buf1612
        del buf1613
        del primals_282
        del primals_283
        # Topologically Sorted Source Nodes: [add__47], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_285, primals_285, 1, stream=stream0)
        del primals_285
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_137.run(primals_288, buf1649, primals_289, buf1650, primals_288, primals_289, 512, stream=stream0)
        del buf1649
        del buf1650
        del primals_288
        del primals_289
        # Topologically Sorted Source Nodes: [add__48], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_291, primals_291, 1, stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_137.run(primals_294, buf1684, primals_295, buf1685, primals_294, primals_295, 512, stream=stream0)
        del buf1684
        del buf1685
        del primals_294
        del primals_295
        # Topologically Sorted Source Nodes: [add__49], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_297, primals_297, 1, stream=stream0)
        del primals_297
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_1_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_138.run(primals_300, buf1719, primals_301, buf1720, primals_300, primals_301, 2048, stream=stream0)
        del buf1719
        del buf1720
        del primals_300
        del primals_301
        # Topologically Sorted Source Nodes: [add__50], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_303, primals_303, 1, stream=stream0)
        del primals_303
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn1], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_137.run(primals_306, buf1755, primals_307, buf1756, primals_306, primals_307, 512, stream=stream0)
        del buf1755
        del buf1756
        del primals_306
        del primals_307
        # Topologically Sorted Source Nodes: [add__51], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_309, primals_309, 1, stream=stream0)
        del primals_309
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn2], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_137.run(primals_312, buf472, primals_313, buf1790, primals_312, primals_313, 512, stream=stream0)
        del buf1790
        del buf472
        del primals_312
        del primals_313
        # Topologically Sorted Source Nodes: [add__52], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__126.run(primals_315, primals_315, 1, stream=stream0)
        del primals_315
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_2_bn3], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_copy__div_mul_sub_138.run(primals_318, buf1584, primals_319, buf1822, primals_318, primals_319, 2048, stream=stream0)
        del buf1584
        del buf1822
        del primals_318
        del primals_319
    return (buf1853, primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, buf1, buf2, buf3, buf4, buf18, buf25, buf26, buf27, buf33, reinterpret_tensor(buf38, (3211264, 16), (16, 1), 0), buf41, buf42, buf44, buf45, buf46, buf47, buf58, buf65, buf66, buf67, reinterpret_tensor(buf77, (802816, 16), (16, 1), 0), buf79, buf81, buf82, buf83, buf84, buf95, buf102, buf103, buf104, reinterpret_tensor(buf112, (802816, 16), (16, 1), 0), buf114, buf116, buf117, buf118, buf119, buf131, buf138, buf139, buf140, buf144, buf145, buf146, buf147, buf158, buf165, buf166, buf167, reinterpret_tensor(buf177, (3211264, 16), (16, 1), 0), buf179, buf181, buf182, buf183, buf184, buf195, buf202, buf203, buf204, reinterpret_tensor(buf212, (802816, 16), (16, 1), 0), buf214, buf216, buf217, buf218, buf219, buf230, buf237, buf238, buf239, reinterpret_tensor(buf247, (802816, 16), (16, 1), 0), buf249, buf251, buf252, buf253, buf254, buf265, buf272, buf273, buf274, reinterpret_tensor(buf283, (3211264, 16), (16, 1), 0), buf285, buf287, buf288, buf289, buf290, buf301, buf308, buf309, buf310, reinterpret_tensor(buf318, (802816, 16), (16, 1), 0), buf320, buf322, buf323, buf324, buf325, buf335, buf342, buf343, buf344, reinterpret_tensor(buf351, (802816, 16), (16, 1), 0), buf353, buf355, buf356, buf357, buf358, buf369, buf376, buf377, buf378, buf37, buf387, buf389, buf390, buf391, buf392, buf404, buf411, buf412, buf413, reinterpret_tensor(buf422, (1605632, 16), (16, 1), 0), buf424, buf426, buf427, buf428, buf429, buf440, buf447, buf448, buf449, reinterpret_tensor(buf459, (401408, 16), (16, 1), 0), buf461, buf463, buf464, buf465, buf466, buf478, buf485, buf486, buf487, buf491, buf492, buf493, buf494, buf505, buf512, buf513, buf514, reinterpret_tensor(buf524, (1605632, 16), (16, 1), 0), buf526, buf528, buf529, buf530, buf531, buf542, buf549, buf550, buf551, reinterpret_tensor(buf559, (401408, 16), (16, 1), 0), buf561, buf563, buf564, buf565, buf566, buf577, buf584, buf585, buf586, reinterpret_tensor(buf594, (401408, 16), (16, 1), 0), buf596, buf598, buf599, buf600, buf601, buf612, buf619, buf620, buf621, reinterpret_tensor(buf630, (1605632, 16), (16, 1), 0), buf632, buf634, buf635, buf636, buf637, buf648, buf655, buf656, buf657, reinterpret_tensor(buf665, (401408, 16), (16, 1), 0), buf667, buf669, buf670, buf671, buf672, buf683, buf690, buf691, buf692, reinterpret_tensor(buf700, (401408, 16), (16, 1), 0), buf702, buf704, buf705, buf706, buf707, buf718, buf725, buf726, buf727, reinterpret_tensor(buf736, (1605632, 16), (16, 1), 0), buf738, buf740, buf741, buf742, buf743, buf754, buf761, buf762, buf763, reinterpret_tensor(buf771, (401408, 16), (16, 1), 0), buf773, buf775, buf776, buf777, buf778, buf788, buf795, buf796, buf797, reinterpret_tensor(buf804, (401408, 16), (16, 1), 0), buf806, buf808, buf809, buf810, buf811, buf822, buf829, buf830, buf831, buf421, buf840, buf842, buf843, buf844, buf845, buf856, buf863, buf864, buf865, reinterpret_tensor(buf873, (802816, 16), (16, 1), 0), buf875, buf877, buf878, buf879, buf880, buf891, buf898, buf899, buf900, reinterpret_tensor(buf910, (200704, 16), (16, 1), 0), buf912, buf914, buf915, buf916, buf917, buf929, buf936, buf937, buf938, buf942, buf943, buf944, buf945, buf956, buf963, buf964, buf965, reinterpret_tensor(buf975, (802816, 16), (16, 1), 0), buf977, buf979, buf980, buf981, buf982, buf993, buf1000, buf1001, buf1002, reinterpret_tensor(buf1010, (200704, 16), (16, 1), 0), buf1012, buf1014, buf1015, buf1016, buf1017, buf1028, buf1035, buf1036, buf1037, reinterpret_tensor(buf1045, (200704, 16), (16, 1), 0), buf1047, buf1049, buf1050, buf1051, buf1052, buf1063, buf1070, buf1071, buf1072, reinterpret_tensor(buf1081, (802816, 16), (16, 1), 0), buf1083, buf1085, buf1086, buf1087, buf1088, buf1099, buf1106, buf1107, buf1108, reinterpret_tensor(buf1116, (200704, 16), (16, 1), 0), buf1118, buf1120, buf1121, buf1122, buf1123, buf1134, buf1141, buf1142, buf1143, reinterpret_tensor(buf1151, (200704, 16), (16, 1), 0), buf1153, buf1155, buf1156, buf1157, buf1158, buf1169, buf1176, buf1177, buf1178, reinterpret_tensor(buf1187, (802816, 16), (16, 1), 0), buf1189, buf1191, buf1192, buf1193, buf1194, buf1205, buf1212, buf1213, buf1214, reinterpret_tensor(buf1222, (200704, 16), (16, 1), 0), buf1224, buf1226, buf1227, buf1228, buf1229, buf1240, buf1247, buf1248, buf1249, reinterpret_tensor(buf1257, (200704, 16), (16, 1), 0), buf1259, buf1261, buf1262, buf1263, buf1264, buf1275, buf1282, buf1283, buf1284, reinterpret_tensor(buf1293, (802816, 16), (16, 1), 0), buf1295, buf1297, buf1298, buf1299, buf1300, buf1311, buf1318, buf1319, buf1320, reinterpret_tensor(buf1328, (200704, 16), (16, 1), 0), buf1330, buf1332, buf1333, buf1334, buf1335, buf1346, buf1353, buf1354, buf1355, reinterpret_tensor(buf1363, (200704, 16), (16, 1), 0), buf1365, buf1367, buf1368, buf1369, buf1370, buf1381, buf1388, buf1389, buf1390, reinterpret_tensor(buf1399, (802816, 16), (16, 1), 0), buf1401, buf1403, buf1404, buf1405, buf1406, buf1417, buf1424, buf1425, buf1426, reinterpret_tensor(buf1434, (200704, 16), (16, 1), 0), buf1436, buf1438, buf1439, buf1440, buf1441, buf1451, buf1458, buf1459, buf1460, buf909, buf1468, buf1470, buf1471, buf1472, buf1473, buf1483, buf1490, buf1491, buf1492, buf76, buf1501, buf1503, buf1504, buf1505, buf1506, buf1517, buf1524, buf1525, buf1526, reinterpret_tensor(buf1534, (401408, 16), (16, 1), 0), buf1536, buf1538, buf1539, buf1540, buf1541, buf1552, buf1559, buf1560, buf1561, reinterpret_tensor(buf1571, (100352, 16), (16, 1), 0), buf1573, buf1575, buf1576, buf1577, buf1578, buf1590, buf1597, buf1598, buf1599, buf1603, buf1604, buf1605, buf1606, buf1617, buf1624, buf1625, buf1626, reinterpret_tensor(buf1636, (401408, 16), (16, 1), 0), buf1638, buf1640, buf1641, buf1642, buf1643, buf1654, buf1661, buf1662, buf1663, reinterpret_tensor(buf1671, (100352, 16), (16, 1), 0), buf1673, buf1675, buf1676, buf1677, buf1678, buf1689, buf1696, buf1697, buf1698, reinterpret_tensor(buf1706, (100352, 16), (16, 1), 0), buf1708, buf1710, buf1711, buf1712, buf1713, buf1724, buf1731, buf1732, buf1733, reinterpret_tensor(buf1742, (401408, 16), (16, 1), 0), buf1744, buf1746, buf1747, buf1748, buf1749, buf1760, buf1767, buf1768, buf1769, reinterpret_tensor(buf1777, (100352, 16), (16, 1), 0), buf1779, buf1781, buf1782, buf1783, buf1784, buf1794, buf1801, buf1802, buf1803, buf1570, buf1811, buf1813, buf1814, buf1815, buf1816, buf1826, buf1833, buf1834, buf1835, buf458, buf1846, buf1847, buf1848, buf1852, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2048, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
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
