# AOT ID: ['0_backward']
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/act_triton_kernel.py:87
dequant_unpack_kernel_0 = async_compile.triton('dequant_unpack_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'dequant_unpack_kernel_0', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'P_ptr': '*i32', 'S_ptr': '*bf16', 'M_ptr': '*bf16', 'Y_ptr': '*bf16', 'stride_p0': 'constexpr', 'stride_p1': 'constexpr', 'stride_y0': 'constexpr', 'stride_y1': 'constexpr', 'BITS': 'constexpr', 'VPW': 'constexpr', 'NWORDS': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def dequant_unpack_kernel(
    P_ptr, S_ptr, M_ptr, Y_ptr,
    stride_p0: tl.constexpr, stride_p1: tl.constexpr,
    stride_y0: tl.constexpr, stride_y1: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr
):
    pid = tl.program_id(0)

    p_block_ptr = tl.make_block_ptr(
        base=P_ptr + pid * stride_p0,
        shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(NWORDS,),
        order=(0,)
    )

    word = tl.load(p_block_ptr)

    scale = tl.load(S_ptr + pid)
    scale_dtype = scale.dtype
    scale = scale.to(tl.float32)
    xmin  = tl.load(M_ptr + pid).to(tl.float32)

    mask = (1 << BITS) - 1
    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    q = ((word[:, None] >> shifts[None, :]) & mask).to(tl.float32)
    q = q * scale + xmin
    q_flat = tl.reshape(q, (NWORDS * VPW,))

    y_block_ptr = tl.make_block_ptr(
        base=Y_ptr + pid * stride_y0,
        shape=(NWORDS * VPW,),
        strides=(stride_y1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    tl.store(y_block_ptr, q_flat.to(scale_dtype))

    # for w in tl.static_range(0, NWORDS):
    #     word = tl.load(P_ptr + pid * stride_p0 + w * stride_p1)
    #     q = (word >> shifts) & mask
    #     y = q.to(tl.float32) * scale + xmin
    #     idx = w * VPW + j
    #     tl.store(Y_ptr + pid * stride_y0 + idx * stride_y1, y)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7e/c7exfzuq4663ivtkh7om5xu2eq3vow4fqewj5fja7ujfrhbu2tgt.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_115 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_1, torch.float32), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5c/c5cuvhizyrqn4rr6s6meylke2rxn7bxkot2x5rajsbjtet7wrdai.py
# Topologically Sorted Source Nodes: [full_257], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_257 => full_default_310
# Graph fragment:
#   %full_default_310 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([401408, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_1 = async_compile.triton('triton_poi_fused_zeros_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jq/cjqpghx77vjcjl5xdb46xfrl55fe2owqcbh4kveez4whw7c3istc.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_142
# Graph fragment:
#   %clone_default_142 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_284,), kwargs = {})
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 616562688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/act_triton_kernel.py:183
unpack_kernel_1 = async_compile.triton('unpack_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'unpack_kernel_1', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'P_ptr': '*i32', 'Y_ptr': '*i8', 'stride_p0': 'constexpr', 'stride_p1': 'constexpr', 'stride_y0': 'constexpr', 'stride_y1': 'constexpr', 'BITS': 'constexpr', 'VPW': 'constexpr', 'NWORDS': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def unpack_kernel(
    P_ptr, Y_ptr,
    stride_p0: tl.constexpr, stride_p1: tl.constexpr,
    stride_y0: tl.constexpr, stride_y1: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr
):
    pid = tl.program_id(0)

    p_block_ptr = tl.make_block_ptr(
        base= P_ptr + pid * stride_p0,
        shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(NWORDS,),
        order=(0,)
    )

    word = tl.load(p_block_ptr)

    mask = (1 << BITS) - 1
    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    q = (word[:, None] >> shifts[None, :]) & mask
    q_flatten = tl.reshape(q, (NWORDS * VPW,))

    y_block_ptr = tl.make_block_ptr(
        base=Y_ptr + pid * stride_y0,
        shape=(NWORDS * VPW,),
        strides=(stride_y1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    tl.store(y_block_ptr, q_flatten.to(tl.int8))

    # mask = (1 << BITS) - 1
    # j = tl.arange(0, VPW)
    # shifts = (j * BITS).to(tl.int32)

    # for w in tl.static_range(0, NWORDS):
    #     word = tl.load(P_ptr + pid * stride_p0 + w * stride_p1)
    #     q = (word >> shifts) & mask
    #     idx = w * VPW + j

    #     tl.store(Y_ptr + pid * stride_y0 + idx * stride_y1,
    #              q)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7f/c7fsx6qkeldk2iygccv6c7er275vvoisk7whihpap2omdbm4h4pg.py
# Topologically Sorted Source Nodes: [layer4_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer4_0_bn3 => full_default_264
# Graph fragment:
#   %full_default_264 : [num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([2048], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_3 = async_compile.triton('triton_poi_fused_zeros_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/f6/cf6j62d6pqiv7troqnswiianrri45or76fdxjllfzpru7djz4jkh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_140, clone_default_141
# Graph fragment:
#   %clone_default_140 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_280,), kwargs = {})
#   %clone_default_141 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_282,), kwargs = {})
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 40960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/rw/crwe76jrnzwczkvje6rcctmyfytm62mn7n5ezg2h5xb3kqw4foaj.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_257, triton_kernel_wrapper_mutation_258
# Graph fragment:
#   %triton_kernel_wrapper_mutation_258 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 436, constant_args_idx: 625, grid: [(2048, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1351, DY: %view_1349, DBETA: %as_strided_default_281, DGAMMA: %as_strided_default_283, M: 100352, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_257 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 438, constant_args_idx: 626, grid: [(2048, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1351, DY: %view_1349, INVSTD: %rsqrt_52, GAMMA: %primals_316, DBETA: %as_strided_default_281, DGAMMA: %as_strided_default_283, DX: %permute_157, M: 100352, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 49
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp1 = 0.02040816326530612
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp5, None)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_2 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_2', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_3 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_3', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/nn/cnncig4g5kitvaii4d277qielp5up35t66qcc64iz4hetldnfq4n.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_106, [None, None, %unsqueeze, %convert_element_type_121]), kwargs = {})
#   %convert_element_type_124 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index, torch.bfloat16), kwargs = {})
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1368, %convert_element_type_124), kwargs = {})
#   %convolution_backward_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_1369, %add_232, %expand_2, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_6 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 7) % 7)
    x0 = (xindex % 7)
    x2 = ((xindex // 49) % 512)
    x3 = xindex // 25088
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.2857142857142857
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 512*tmp9 + 1024*tmp5 + 2048*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yq/cyqi7ypck54yyup5wyr7iu7xdzu3lr5j5uqjbgz4bvho36ukreuv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_125 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_693, torch.float32), kwargs = {})
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2i/c2iidtilnivhtl3dx63mtbrd64w7s5l2oxhih7pkffxdm243yj7l.py
# Topologically Sorted Source Nodes: [full_260], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_260 => full_default_313
# Graph fragment:
#   %full_default_313 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([100352, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_8 = async_compile.triton('triton_poi_fused_zeros_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4o/c4orejg4cx5m5rmpduuvmtn3smflfdhuqc2z43p66ugifq4fgk6n.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_136, clone_default_139
# Graph fragment:
#   %clone_default_139 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_278,), kwargs = {})
#   %clone_default_136 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_272,), kwargs = {})
triton_poi_fused_9 = async_compile.triton('triton_poi_fused_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256901120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lq/clqmvoh3zv5uevwl7yans244t3rr6zj6fklqmi6d4aqxsg7b4l66.py
# Topologically Sorted Source Nodes: [layer2_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_bn3 => full_default_76
# Graph fragment:
#   %full_default_76 : [num_users=24] = call_function[target=torch.ops.aten.full.default](args = ([512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_10 = async_compile.triton('triton_poi_fused_zeros_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/an/canvyrt23l3xsrplbs7na4xwohb2wundgj4lxoadnq7govogi64p.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_134, clone_default_135, clone_default_137, clone_default_138
# Graph fragment:
#   %clone_default_137 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_274,), kwargs = {})
#   %clone_default_138 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_276,), kwargs = {})
#   %clone_default_134 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_268,), kwargs = {})
#   %clone_default_135 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_270,), kwargs = {})
triton_poi_fused_11 = async_compile.triton('triton_poi_fused_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/62/c62n5ro5b5ulpk5qcako7kzi6de5w262534o2pkpc3fity7skjsz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_252, triton_kernel_wrapper_mutation_253
# Graph fragment:
#   %triton_kernel_wrapper_mutation_253 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 441, constant_args_idx: 630, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1390, DY: %view_1388, DBETA: %as_strided_default_275, DGAMMA: %as_strided_default_277, M: 100352, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_252 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 442, constant_args_idx: 631, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1390, DY: %view_1388, INVSTD: %rsqrt_51, GAMMA: %primals_310, DBETA: %as_strided_default_275, DGAMMA: %as_strided_default_277, DX: %permute_158, M: 100352, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_12 = async_compile.triton('triton_poi_fused_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 102760448, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_12(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 49*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 49*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_4 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_4', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_5 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_5', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bc/cbcjrbkyncqkek4m735ikofr2sjdsjukesz3winlfpddbobxfgbx.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_135 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_705, torch.float32), kwargs = {})
triton_poi_fused__to_copy_13 = async_compile.triton('triton_poi_fused__to_copy_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2w/c2welx5wp4yk27q3w4t64745kyicswof4dwsjqcz7k5yhpuzikns.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_102, [None, None, %unsqueeze, %convert_element_type_121]), kwargs = {})
#   %convert_element_type_144 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_2, torch.bfloat16), kwargs = {})
#   %add_242 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1446, %convert_element_type_144), kwargs = {})
#   %convolution_backward_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_1447, %add_242, %expand_6, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_14 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 7) % 7)
    x0 = (xindex % 7)
    x2 = ((xindex // 49) % 2048)
    x3 = xindex // 100352
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.2857142857142857
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 2048*tmp9 + 4096*tmp5 + 8192*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/tg/ctgjeiawvs5i34gt4wq3befdocxny7isygkrjikgf74ouzlcedrs.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_242, triton_kernel_wrapper_mutation_243
# Graph fragment:
#   %triton_kernel_wrapper_mutation_243 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 449, constant_args_idx: 640, grid: [(2048, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1468, DY: %view_1466, DBETA: %as_strided_default_263, DGAMMA: %as_strided_default_265, M: 100352, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_242 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 450, constant_args_idx: 641, grid: [(2048, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1468, DY: %view_1466, INVSTD: %rsqrt_49, GAMMA: %primals_298, DBETA: %as_strided_default_263, DGAMMA: %as_strided_default_265, DX: %permute_160, M: 100352, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_15 = async_compile.triton('triton_poi_fused_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4194304, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 419430400, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4194304
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y3 = yindex
    x2 = xindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y0 + 2048*x2 + 100352*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.02040816326530612
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tl.store(out_ptr0 + (x2 + 49*y3), tmp10, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 49*y3), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yt/cytluvu3mfpxmebyy23fnxdc6tpdqyk7apsjq2hstaxectt3jz4o.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_118, clone_default_124
# Graph fragment:
#   %clone_default_124 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_248,), kwargs = {})
#   %clone_default_118 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_236,), kwargs = {})
triton_poi_fused_16 = async_compile.triton('triton_poi_fused_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1027604480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xm/cxmzks7ftawjeyem3qx6bcwtiuapcnyzyqi625thunbgvidsz6hj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_159 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %mul_371 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_159, %view_1334), kwargs = {})
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %getitem_713), kwargs = {})
#   %mul_386 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_243, %view_1451), kwargs = {})
#   %add_259 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %getitem_749), kwargs = {})
#   %mul_401 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_259, %view_1568), kwargs = {})
triton_poi_fused_add_div_mul_17 = async_compile.triton('triton_poi_fused_add_div_mul_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4194304, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'in_ptr4': '*bf16', 'in_ptr5': '*i8', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 830472192, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4194304
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y3 = yindex
    x2 = xindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y0 + 2048*x2 + 100352*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (y0 + 2048*x2 + 100352*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.02040816326530612
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tl.store(out_ptr0 + (x2 + 49*y3), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hl/chlptsvfjyxstvj4c33s4ctvxo4rolzhzkugtq7ks46ki7ln4cwh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_121, clone_default_122, clone_default_123
# Graph fragment:
#   %clone_default_122 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_244,), kwargs = {})
#   %clone_default_123 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_246,), kwargs = {})
#   %clone_default_121 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_242,), kwargs = {})
triton_poi_fused_18 = async_compile.triton('triton_poi_fused_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 57344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_18(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7m/c7mxcl4unnkrtatw43p76x3e63nqyagc5qdxt7dtuhnlcuxsatoq.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_88, [None, None, %unsqueeze_6, %convert_element_type_181]), kwargs = {})
#   %convert_element_type_184 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_6, torch.bfloat16), kwargs = {})
#   %add_264 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1602, %convert_element_type_184), kwargs = {})
#   %convolution_backward_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_1603, %add_264, %expand_14, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
#   %add_279 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1715, %convert_element_type_184), kwargs = {})
#   %convolution_backward_19 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_1716, %add_279, %expand_20, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_19 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 14) % 14)
    x0 = (xindex % 14)
    x2 = ((xindex // 196) % 1024)
    x3 = xindex // 200704
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.2857142857142857
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 1024*tmp9 + 4096*tmp5 + 16384*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tmp14 = tmp13 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
    tl.store(out_ptr1 + (x4), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ta/ctakuvpxwlfa33s4ztmfup4vomgxwzsm3djz7iptqe6azccqviro.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_185 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_765, torch.float32), kwargs = {})
triton_poi_fused__to_copy_20 = async_compile.triton('triton_poi_fused__to_copy_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20971520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/te/cteznrvxhbyuh4ikv5qydpgxlxj37d2norq2qxgecnvhdpzasbsr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_90, [None, None, %unsqueeze_6, %convert_element_type_181]), kwargs = {})
#   %convert_element_type_204 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_8, torch.bfloat16), kwargs = {})
#   %add_274 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1676, %convert_element_type_204), kwargs = {})
#   %convolution_backward_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_1677, %add_274, %expand_18, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_21 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 14) % 14)
    x0 = (xindex % 14)
    x2 = ((xindex // 196) % 512)
    x3 = xindex // 100352
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.2857142857142857
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 512*tmp9 + 2048*tmp5 + 8192*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/kv/ckvxxg7ai7uldjw6reo2ribbydkyz7frttiocfs33gcu3233zptm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_213, triton_kernel_wrapper_mutation_214
# Graph fragment:
#   %triton_kernel_wrapper_mutation_214 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 473, constant_args_idx: 669, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1698, DY: %view_1696, DBETA: %as_strided_default_233, DGAMMA: %as_strided_default_235, M: 401408, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_213 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 474, constant_args_idx: 670, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1698, DY: %view_1696, INVSTD: %rsqrt_43, GAMMA: %primals_262, DBETA: %as_strided_default_233, DGAMMA: %as_strided_default_235, DX: %permute_166, M: 401408, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_22 = async_compile.triton('triton_poi_fused_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_22(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 196*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_6 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_6', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_7 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_7', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/s4/cs42ddmrqku4kpy52q2xpjqvni3dyqpvmyw7ze5wywfpinmr3j63.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_215 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_800, torch.float32), kwargs = {})
triton_poi_fused__to_copy_23 = async_compile.triton('triton_poi_fused__to_copy_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5242880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/m3/cm3fuueftdlzssluzolyymqb7abzidimjlhpl6u5zoljyefrorvd.py
# Topologically Sorted Source Nodes: [full_286], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_286 => full_default_339
# Graph fragment:
#   %full_default_339 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([802816, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_24 = async_compile.triton('triton_poi_fused_zeros_24', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_24(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ed/cedm3dfuun6ttgygemqvesfcwlocli3upolriy2cjfmnaj7dnnmy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_115
# Graph fragment:
#   %clone_default_115 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_230,), kwargs = {})
triton_poi_fused_25 = async_compile.triton('triton_poi_fused_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1233125376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/si/csimszbzlnibhfkkhgnxx45hltajv235iqrhbuyf52r3onhoulp7.py
# Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer3_0_bn3 => full_default_152
# Graph fragment:
#   %full_default_152 : [num_users=16] = call_function[target=torch.ops.aten.full.default](args = ([1024], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_26 = async_compile.triton('triton_poi_fused_zeros_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_26(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/r6/cr642vrmfdjryxahlzysodlfkbqwzpb7agg36jhjrn5wopj7riu7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_113, clone_default_114
# Graph fragment:
#   %clone_default_113 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_226,), kwargs = {})
#   %clone_default_114 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_228,), kwargs = {})
triton_poi_fused_27 = async_compile.triton('triton_poi_fused_27', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_27(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vz/cvzhlismqejdkazzneyn2dbomoeaagn7o4qyppoazet3svm26ia7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_208, triton_kernel_wrapper_mutation_209
# Graph fragment:
#   %triton_kernel_wrapper_mutation_209 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 477, constant_args_idx: 674, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1737, DY: %view_1735, DBETA: %as_strided_default_227, DGAMMA: %as_strided_default_229, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_208 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 478, constant_args_idx: 675, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1737, DY: %view_1735, INVSTD: %rsqrt_42, GAMMA: %primals_256, DBETA: %as_strided_default_227, DGAMMA: %as_strided_default_229, DX: %permute_167, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_28 = async_compile.triton('triton_poi_fused_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_28(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 196*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_8 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_8', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_9 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_9', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mm/cmmps7hticleboruhfv2bym4iuxsiawgm4ozru42jf4ruydnnji5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_86, [None, None, %unsqueeze_6, %convert_element_type_181]), kwargs = {})
#   %convert_element_type_224 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_10, torch.bfloat16), kwargs = {})
#   %add_285 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1754, %convert_element_type_224), kwargs = {})
#   %convolution_backward_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_1755, %add_285, %expand_22, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 14) % 14)
    x0 = (xindex % 14)
    x2 = ((xindex // 196) % 256)
    x3 = xindex // 50176
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.2857142857142857
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 256*tmp9 + 1024*tmp5 + 4096*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ui/cuimlhmj4bkmfgbw7l24ua6i6g2zjjksjnewzwrebup7ef6gsphf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_225 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_812, torch.float32), kwargs = {})
triton_poi_fused__to_copy_30 = async_compile.triton('triton_poi_fused__to_copy_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2621440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ds/cds4ujvsri7nmjs3mpna2mkn5pdhl2f46riy32wuugifqzvoybsw.py
# Topologically Sorted Source Nodes: [full_289], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_289 => full_default_342
# Graph fragment:
#   %full_default_342 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([200704, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_31 = async_compile.triton('triton_poi_fused_zeros_31', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_31(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/j7/cj7epf6bynt3h77mg22tyw2awsryghoapb5ty4lbcqnb3gvrcxyz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_109, clone_default_112
# Graph fragment:
#   %clone_default_112 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_224,), kwargs = {})
#   %clone_default_109 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_218,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 513802240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_32(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/np/cnpdkcemzekmzsvulfv3m72z7al6irycgopg24j27v6keivg4cdd.py
# Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer1_0_bn3 => full_default_18
# Graph fragment:
#   %full_default_18 : [num_users=34] = call_function[target=torch.ops.aten.full.default](args = ([256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_33 = async_compile.triton('triton_poi_fused_zeros_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_33(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ud/cud26pmur3eowjkbapuxlcfu5fnuqvrg76wqjngi6gqcjf5e76lx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_107, clone_default_108, clone_default_110, clone_default_111
# Graph fragment:
#   %clone_default_110 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_220,), kwargs = {})
#   %clone_default_111 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_222,), kwargs = {})
#   %clone_default_107 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_214,), kwargs = {})
#   %clone_default_108 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_216,), kwargs = {})
triton_poi_fused_34 = async_compile.triton('triton_poi_fused_34', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_34(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/5y/c5yzjkmsopmtojjqfal3t2z2nvkjqxzxrcnvmi2ph42psleos2me.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_203, triton_kernel_wrapper_mutation_204
# Graph fragment:
#   %triton_kernel_wrapper_mutation_204 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 481, constant_args_idx: 679, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1776, DY: %view_1774, DBETA: %as_strided_default_221, DGAMMA: %as_strided_default_223, M: 401408, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_203 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 482, constant_args_idx: 680, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1776, DY: %view_1774, INVSTD: %rsqrt_41, GAMMA: %primals_250, DBETA: %as_strided_default_221, DGAMMA: %as_strided_default_223, DX: %permute_168, M: 401408, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_35 = async_compile.triton('triton_poi_fused_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_35(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 196*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_10 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_10', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_11 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_11', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sr/csr7zd6n5y34lytfh6gotgfw6tlzegzdur4vylpcmfosbugsx5jk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_235 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_824, torch.float32), kwargs = {})
triton_poi_fused__to_copy_36 = async_compile.triton('triton_poi_fused__to_copy_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5898240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xn/cxngzgd6ozjwn5m6jjhkivqlfls3a5ljkdktvry5vq55q7zzxia6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_12 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_82, [None, None, %unsqueeze_6, %convert_element_type_181]), kwargs = {})
#   %convert_element_type_244 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_12, torch.bfloat16), kwargs = {})
#   %add_295 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1832, %convert_element_type_244), kwargs = {})
#   %convolution_backward_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_1833, %add_295, %expand_26, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_37 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_37(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 14) % 14)
    x0 = (xindex % 14)
    x2 = ((xindex // 196) % 1024)
    x3 = xindex // 200704
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.2857142857142857
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 1024*tmp9 + 4096*tmp5 + 16384*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lw/clwlcoxkrdy4jkta4jvbfv4jc74ob37sha544yciqkhngwdih366.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_106, clone_default_97
# Graph fragment:
#   %clone_default_106 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_212,), kwargs = {})
#   %clone_default_97 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_194,), kwargs = {})
triton_poi_fused_38 = async_compile.triton('triton_poi_fused_38', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2055208960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_38(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/am/cammiwbrtkikv7ekm66qnxkpbfghl2kchb6qps2odrvl7zgy5t4a.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_280 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_761, %getitem_796), kwargs = {})
#   %mul_420 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_280, %view_1720), kwargs = {})
#   %add_296 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_420, %getitem_832), kwargs = {})
#   %mul_435 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_296, %view_1837), kwargs = {})
triton_poi_fused_add_mul_39 = async_compile.triton('triton_poi_fused_add_mul_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'in_ptr3': '*bf16', 'in_ptr4': '*i8', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2466250752, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tl.store(out_ptr0 + (x2 + 196*y3), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hh/chhdkfnr25zsnm4g6fuujhbffha5o6xffwi6qi2o4vmeaesgctbq.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_104, clone_default_105, clone_default_95, clone_default_96
# Graph fragment:
#   %clone_default_104 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_208,), kwargs = {})
#   %clone_default_105 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_210,), kwargs = {})
#   %clone_default_95 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_190,), kwargs = {})
#   %clone_default_96 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_192,), kwargs = {})
triton_poi_fused_40 = async_compile.triton('triton_poi_fused_40', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_40(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/37/c37f3i7bij7pnxw5gqxytc2zhfqkwbw7hjri5fsyezdwp7s4lmvy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_178, triton_kernel_wrapper_mutation_179
# Graph fragment:
#   %triton_kernel_wrapper_mutation_179 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 501, constant_args_idx: 704, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1971, DY: %view_1969, DBETA: %as_strided_default_191, DGAMMA: %as_strided_default_193, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_178 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 502, constant_args_idx: 705, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1971, DY: %view_1969, INVSTD: %rsqrt_36, GAMMA: %primals_220, DBETA: %as_strided_default_191, DGAMMA: %as_strided_default_193, DX: %permute_173, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_41 = async_compile.triton('triton_poi_fused_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 4110417920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_41(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 196*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/56/c56ayxjp34us7fbzhp2nnxhrnhgk2lww2rvuxfstt4w2fxgbj2rd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_312 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_435, %getitem_868), kwargs = {})
#   %mul_450 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_312, %view_1954), kwargs = {})
#   %add_328 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_450, %getitem_904), kwargs = {})
#   %mul_465 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_328, %view_2071), kwargs = {})
triton_poi_fused_add_mul_42 = async_compile.triton('triton_poi_fused_add_mul_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 2466250752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 196*y3), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/e4/ce4dcr4ygroh3peb6tsaeaitbuwc4ayqwgyy22i7n72byzcdbpxs.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_129, triton_kernel_wrapper_mutation_130, triton_kernel_wrapper_mutation_133, triton_kernel_wrapper_mutation_134
# Graph fragment:
#   %add_344 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_465, %getitem_940), kwargs = {})
#   %mul_480 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_344, %view_2188), kwargs = {})
#   %add_360 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_480, %getitem_976), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_360, %view_2305), kwargs = {})
#   %triton_kernel_wrapper_mutation_134 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 537, constant_args_idx: 749, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2322, DY: %view_2320, DBETA: %as_strided_default_137, DGAMMA: %as_strided_default_139, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_133 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 538, constant_args_idx: 750, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2322, DY: %view_2320, INVSTD: %rsqrt_27, GAMMA: %primals_166, DBETA: %as_strided_default_137, DGAMMA: %as_strided_default_139, DX: %permute_182, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_130 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 541, constant_args_idx: 753, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2357, DY: %view_2320, DBETA: %full_default_152, DGAMMA: %as_strided_default_135, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_129 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 542, constant_args_idx: 754, grid: [(1024, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2357, DY: %view_2320, INVSTD: %rsqrt_26, GAMMA: %primals_160, DBETA: %full_default_152, DGAMMA: %as_strided_default_135, DX: %permute_183, M: 401408, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_add_mul_43 = async_compile.triton('triton_poi_fused_add_mul_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 7398752256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tl.store(out_ptr0 + (x2 + 196*y3), tmp10, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp10, xmask & ymask)
    tl.store(out_ptr2 + (x2 + 196*y3), tmp10, xmask & ymask)
    tl.store(out_ptr3 + (x2 + 196*y3), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jy/cjydhyyqu4eqfox64t2qopakn7brdeepx3hdmkrijmbkaavf676i.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_67, clone_default_68, clone_default_69
# Graph fragment:
#   %clone_default_68 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_136,), kwargs = {})
#   %clone_default_69 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_138,), kwargs = {})
#   %clone_default_67 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_134,), kwargs = {})
triton_poi_fused_44 = async_compile.triton('triton_poi_fused_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 28672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_44(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
    tl.store(out_ptr2 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ao/cao7366dwwpmbjvkfqhx6wsadhwhgecxj32bon4hn6hrerla7gjv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_25 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_50, [None, None, %unsqueeze_25, %convert_element_type_371]), kwargs = {})
#   %convert_element_type_374 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_25, torch.bfloat16), kwargs = {})
#   %add_365 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2339, %convert_element_type_374), kwargs = {})
#   %convolution_backward_51 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_2340, %add_365, %expand_52, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
#   %add_380 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2452, %convert_element_type_374), kwargs = {})
#   %convolution_backward_57 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_2453, %add_380, %expand_58, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_45 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_45(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 822083584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 28) % 28)
    x0 = (xindex % 28)
    x2 = ((xindex // 784) % 512)
    x3 = xindex // 401408
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.32142857142857145
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 512*tmp9 + 4608*tmp5 + 41472*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tmp14 = tmp13 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
    tl.store(out_ptr1 + (x4), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/m6/cm6n4mdvwfceki5cqw2eaxodps6b72bmmcxwmfrfzvt6v5syawh2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_27 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_52, [None, None, %unsqueeze_25, %convert_element_type_371]), kwargs = {})
#   %convert_element_type_394 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_27, torch.bfloat16), kwargs = {})
#   %add_375 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2413, %convert_element_type_394), kwargs = {})
#   %convolution_backward_55 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_2414, %add_375, %expand_56, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_46 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_46(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 28) % 28)
    x0 = (xindex % 28)
    x2 = ((xindex // 784) % 256)
    x3 = xindex // 200704
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.32142857142857145
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 256*tmp9 + 2304*tmp5 + 20736*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/j3/cj3hkgxmd46gbmjzo2wzqexpzfnoqa45sprcv2mlm4ub2a7wa652.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_119, triton_kernel_wrapper_mutation_120
# Graph fragment:
#   %triton_kernel_wrapper_mutation_120 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 549, constant_args_idx: 763, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2435, DY: %view_2433, DBETA: %as_strided_default_125, DGAMMA: %as_strided_default_127, M: 1605632, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_119 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 550, constant_args_idx: 764, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2435, DY: %view_2433, INVSTD: %rsqrt_24, GAMMA: %primals_148, DBETA: %as_strided_default_125, DGAMMA: %as_strided_default_127, DX: %permute_185, M: 1605632, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_47 = async_compile.triton('triton_poi_fused_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_47(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 784*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_12 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_12', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_13 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_13', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4i/c4ifiijkuv4lvjre35izfus4weuweek34dscdjh65zdh6bzkp672.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_405 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1027, torch.float32), kwargs = {})
triton_poi_fused__to_copy_48 = async_compile.triton('triton_poi_fused__to_copy_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1310720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ke/ckeacmotd3qiv5kanttfrmvuas72odfzisj7cpvrmfrcuneywt3s.py
# Topologically Sorted Source Nodes: [full_342], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_342 => full_default_395
# Graph fragment:
#   %full_default_395 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([1605632, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_49 = async_compile.triton('triton_poi_fused_zeros_49', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_49(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 822083584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/v6/cv6ieyjk2celbdhsk6rmwbvdzuvy53jcod6gumjp6cv7zi2zg24e.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_61
# Graph fragment:
#   %clone_default_61 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_122,), kwargs = {})
triton_poi_fused_50 = async_compile.triton('triton_poi_fused_50', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2466250752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_50(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 822083584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/di/cdios4oeddatyj2jr4x4mxw6mwrxebnq2lnvhvniu5nsq77fb4na.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_59, clone_default_60
# Graph fragment:
#   %clone_default_59 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_118,), kwargs = {})
#   %clone_default_60 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_120,), kwargs = {})
triton_poi_fused_51 = async_compile.triton('triton_poi_fused_51', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_51(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wi/cwijwjg4fx7z2t45epmjqr2frihqnbkgnwi3ieb4msosmrsoowqv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_114, triton_kernel_wrapper_mutation_115
# Graph fragment:
#   %triton_kernel_wrapper_mutation_115 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 553, constant_args_idx: 768, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2474, DY: %view_2472, DBETA: %as_strided_default_119, DGAMMA: %as_strided_default_121, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_114 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 554, constant_args_idx: 769, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2474, DY: %view_2472, INVSTD: %rsqrt_23, GAMMA: %primals_142, DBETA: %as_strided_default_119, DGAMMA: %as_strided_default_121, DX: %permute_186, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_52 = async_compile.triton('triton_poi_fused_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3288334336, 'x': 6576668672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_52(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 512*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 784*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_14 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_14', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_15 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_15', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xj/cxjjm3pdscyiqq3xtyjuh2ajq27ltio6gcm7kgqmuvrbkrvlfpwd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_29 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_48, [None, None, %unsqueeze_25, %convert_element_type_371]), kwargs = {})
#   %convert_element_type_414 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_29, torch.bfloat16), kwargs = {})
#   %add_386 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2491, %convert_element_type_414), kwargs = {})
#   %convolution_backward_59 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_2492, %add_386, %expand_60, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 28) % 28)
    x0 = (xindex % 28)
    x2 = ((xindex // 784) % 128)
    x3 = xindex // 100352
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.32142857142857145
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 128*tmp9 + 1152*tmp5 + 10368*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ko/ckogdp2botu5l2zpqkzdx74i7xh6b5c7uutptfxy7bbjz2lxhz6g.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_415 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1039, torch.float32), kwargs = {})
triton_poi_fused__to_copy_54 = async_compile.triton('triton_poi_fused__to_copy_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 655360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_54(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/zm/czmd3o5xpbum2hchhgmrvx6s4ujtzyivs4gshrcj64vy7vfcjonf.py
# Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_bn1 => full_default_64
# Graph fragment:
#   %full_default_64 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_55 = async_compile.triton('triton_poi_fused_zeros_55', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_55(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ko/ckopc4nf625qfndzdis7ux3soseqwnl3fmaxm423efscec7ex4zp.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_53, clone_default_54, clone_default_56, clone_default_57
# Graph fragment:
#   %clone_default_56 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_112,), kwargs = {})
#   %clone_default_57 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_114,), kwargs = {})
#   %clone_default_53 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_106,), kwargs = {})
#   %clone_default_54 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_108,), kwargs = {})
triton_poi_fused_56 = async_compile.triton('triton_poi_fused_56', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_56(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/hg/chgyu5rt6bry4knhlimzxbqk67b6r36hjrg62rabtcfmerqup2gl.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_109, triton_kernel_wrapper_mutation_110
# Graph fragment:
#   %triton_kernel_wrapper_mutation_110 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 557, constant_args_idx: 773, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2513, DY: %view_2511, DBETA: %as_strided_default_113, DGAMMA: %as_strided_default_115, M: 1605632, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_109 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 558, constant_args_idx: 774, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2513, DY: %view_2511, INVSTD: %rsqrt_22, GAMMA: %primals_136, DBETA: %as_strided_default_113, DGAMMA: %as_strided_default_115, DX: %permute_187, M: 1605632, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_57 = async_compile.triton('triton_poi_fused_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_57(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 784*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_16 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_16', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_17 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_17', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/nh/cnhs7nhak7ubaqwdohpb4q56akkcyaorfnhpvfqrtcoaglc5xlue.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_425 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1051, torch.float32), kwargs = {})
triton_poi_fused__to_copy_58 = async_compile.triton('triton_poi_fused__to_copy_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1474560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_58(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mj/cmjnbqc7ibnnckee5zeupp3resyo3hwgxygsfbvxohmpq5x2pfti.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_31 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_44, [None, None, %unsqueeze_25, %convert_element_type_371]), kwargs = {})
#   %convert_element_type_434 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_31, torch.bfloat16), kwargs = {})
#   %add_396 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2569, %convert_element_type_434), kwargs = {})
#   %convolution_backward_63 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_2570, %add_396, %expand_64, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_59 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_59(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 822083584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 28) % 28)
    x0 = (xindex % 28)
    x2 = ((xindex // 784) % 512)
    x3 = xindex // 401408
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.32142857142857145
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 512*tmp9 + 4608*tmp5 + 41472*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ql/cqldvjq774wcwobxm4eze53bh4kze7yyi3omidt4k7j7ahtwd6lc.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_43, clone_default_52
# Graph fragment:
#   %clone_default_52 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_104,), kwargs = {})
#   %clone_default_43 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_86,), kwargs = {})
triton_poi_fused_60 = async_compile.triton('triton_poi_fused_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4110417920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_60(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 822083584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7s/c7s5sd4bb5k6ormzkwdbipp3f46lchhgh4yedujirnmrgru2srpw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_381 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_988, %getitem_1023), kwargs = {})
#   %mul_514 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_381, %view_2457), kwargs = {})
#   %add_397 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_514, %getitem_1059), kwargs = {})
#   %mul_529 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_397, %view_2574), kwargs = {})
triton_poi_fused_add_mul_61 = async_compile.triton('triton_poi_fused_add_mul_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'in_ptr3': '*bf16', 'in_ptr4': '*i8', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4932501504, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 512*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + 512*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tl.store(out_ptr0 + (x2 + 784*y3), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/6g/c6g5mqs7cungqilyber7wojj757myhgpnvwcm4xqsmpaeije7smy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_84, triton_kernel_wrapper_mutation_85
# Graph fragment:
#   %triton_kernel_wrapper_mutation_85 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 577, constant_args_idx: 798, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2708, DY: %view_2706, DBETA: %as_strided_default_83, DGAMMA: %as_strided_default_85, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_84 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 578, constant_args_idx: 799, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2708, DY: %view_2706, INVSTD: %rsqrt_17, GAMMA: %primals_106, DBETA: %as_strided_default_83, DGAMMA: %as_strided_default_85, DX: %permute_192, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_62 = async_compile.triton('triton_poi_fused_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 8220835840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_62(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 512*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 784*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bj/cbjza5kslap6k4zy54hxknjvmuv2dwr74zqe6eilc5jndf23fxwi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_65, triton_kernel_wrapper_mutation_66, triton_kernel_wrapper_mutation_69, triton_kernel_wrapper_mutation_70
# Graph fragment:
#   %add_413 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_529, %getitem_1095), kwargs = {})
#   %mul_544 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_413, %view_2691), kwargs = {})
#   %add_429 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_544, %getitem_1131), kwargs = {})
#   %mul_559 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_429, %view_2808), kwargs = {})
#   %triton_kernel_wrapper_mutation_70 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 589, constant_args_idx: 813, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2825, DY: %view_2823, DBETA: %as_strided_default_65, DGAMMA: %as_strided_default_67, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_69 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 590, constant_args_idx: 814, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2825, DY: %view_2823, INVSTD: %rsqrt_14, GAMMA: %primals_88, DBETA: %as_strided_default_65, DGAMMA: %as_strided_default_67, DX: %permute_195, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_66 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 593, constant_args_idx: 817, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2860, DY: %view_2823, DBETA: %full_default_76, DGAMMA: %as_strided_default_63, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_65 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 594, constant_args_idx: 818, grid: [(512, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2860, DY: %view_2823, INVSTD: %rsqrt_13, GAMMA: %primals_82, DBETA: %full_default_76, DGAMMA: %as_strided_default_63, DX: %permute_196, M: 1605632, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_add_mul_63 = async_compile.triton('triton_poi_fused_add_mul_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3288334336, 'x': 14797504512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_63(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (y0 + 512*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y0 + 512*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tl.store(out_ptr0 + (x2 + 784*y3), tmp10, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp10, xmask & ymask)
    tl.store(out_ptr2 + (x2 + 784*y3), tmp10, xmask & ymask)
    tl.store(out_ptr3 + (x2 + 784*y3), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bb/cbbjmgs6bf5vwrxc3pcq2u7vcctyvv2izgc24prrehxy6rkxhfzu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_31, clone_default_32, clone_default_33
# Graph fragment:
#   %clone_default_32 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_64,), kwargs = {})
#   %clone_default_33 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_66,), kwargs = {})
#   %clone_default_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_62,), kwargs = {})
triton_poi_fused_64 = async_compile.triton('triton_poi_fused_64', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_64(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/ar/carmmgq7nx2es2tkdu3kefctookuljyd6rv3tveee34zlmszlkp6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_38 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_24, [None, None, %unsqueeze_38, %convert_element_type_501]), kwargs = {})
#   %convert_element_type_504 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_38, torch.bfloat16), kwargs = {})
#   %add_434 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2842, %convert_element_type_504), kwargs = {})
#   %convolution_backward_77 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_2843, %add_434, %expand_78, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
#   %add_449 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2955, %convert_element_type_504), kwargs = {})
#   %convolution_backward_83 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_2956, %add_449, %expand_84, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_65 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_65(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 56) % 56)
    x0 = (xindex % 56)
    x2 = ((xindex // 3136) % 256)
    x3 = xindex // 802816
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.32142857142857145
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 256*tmp9 + 4608*tmp5 + 82944*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tmp14 = tmp13 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
    tl.store(out_ptr1 + (x4), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/v2/cv2zqsh7hmirzb7et3p5scfhqel35pv7shbic3q3nt4vyfgvehrs.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_28, clone_default_29, clone_default_30
# Graph fragment:
#   %clone_default_29 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_58,), kwargs = {})
#   %clone_default_30 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_60,), kwargs = {})
#   %clone_default_28 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_56,), kwargs = {})
triton_poi_fused_66 = async_compile.triton('triton_poi_fused_66', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_66(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/gj/cgj63vgekvprhieadqbe7h5yro4xpbplehkqgda6w2i7mpv6ljsp.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_40 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_26, [None, None, %unsqueeze_38, %convert_element_type_501]), kwargs = {})
#   %convert_element_type_524 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_40, torch.bfloat16), kwargs = {})
#   %add_444 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2916, %convert_element_type_524), kwargs = {})
#   %convolution_backward_81 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_2917, %add_444, %expand_82, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_67 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_67(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 822083584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 56) % 56)
    x0 = (xindex % 56)
    x2 = ((xindex // 3136) % 128)
    x3 = xindex // 401408
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.32142857142857145
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 128*tmp9 + 2304*tmp5 + 41472*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/go/cgoinn5g7tbavjugvxo2mh27ychtawpvxayj6cae22g5gjgzf7ty.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_55, triton_kernel_wrapper_mutation_56
# Graph fragment:
#   %triton_kernel_wrapper_mutation_56 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 601, constant_args_idx: 827, grid: [(128, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2938, DY: %view_2936, DBETA: %full_default_64, DGAMMA: %as_strided_default_57, M: 6422528, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_55 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 602, constant_args_idx: 828, grid: [(128, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2938, DY: %view_2936, INVSTD: %rsqrt_11, GAMMA: %primals_70, DBETA: %full_default_64, DGAMMA: %as_strided_default_57, DX: %permute_198, M: 6422528, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_68 = async_compile.triton('triton_poi_fused_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 6576668672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_68(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_18 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_18', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_19 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_19', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sr/csrpqfsla477auwj7j4wqgywn5eiyv7mpegs6btxdbukyyp3cfjb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_535 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1182, torch.float32), kwargs = {})
triton_poi_fused__to_copy_69 = async_compile.triton('triton_poi_fused__to_copy_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 327680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_69(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5s/c5sz3f7jae4opbhw4tupuztvnotucym4vdeijcbbwuxee2suvmfb.py
# Topologically Sorted Source Nodes: [full_380], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_380 => full_default_433
# Graph fragment:
#   %full_default_433 : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([3211264, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_70 = async_compile.triton('triton_poi_fused_zeros_70', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_70(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/46/c46snmfstxmziv4rb47ntdfecypnqp4qszdku6tsibvdesfyk2x5.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_27
# Graph fragment:
#   %clone_default_27 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_54,), kwargs = {})
triton_poi_fused_71 = async_compile.triton('triton_poi_fused_71', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_71', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4932501504}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_71(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/r7/cr7k2k4u3b62gev2h75w5vm4f54cxsph7xem25dpgaguxbulwwxa.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_25, clone_default_26
# Graph fragment:
#   %clone_default_25 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_50,), kwargs = {})
#   %clone_default_26 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_52,), kwargs = {})
triton_poi_fused_72 = async_compile.triton('triton_poi_fused_72', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_72', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_72(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/rk/crkn24ey3z7saoddwqgfxudcbvxlsdkjq7xvsqajo62lsj2dukkw.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_50, triton_kernel_wrapper_mutation_51
# Graph fragment:
#   %triton_kernel_wrapper_mutation_51 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 605, constant_args_idx: 832, grid: [(256, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2977, DY: %view_2975, DBETA: %as_strided_default_51, DGAMMA: %as_strided_default_53, M: 6422528, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_50 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 606, constant_args_idx: 833, grid: [(256, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2977, DY: %view_2975, INVSTD: %rsqrt_10, GAMMA: %primals_64, DBETA: %as_strided_default_51, DGAMMA: %as_strided_default_53, DX: %permute_199, M: 6422528, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_73 = async_compile.triton('triton_poi_fused_73', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_73', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 6576668672, 'x': 13153337344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_73(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 256*x2 + 802816*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_20 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_20', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_21 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_21', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/x6/cx6enfwsiluptlw3ju5u23zajaqpldocqgnv4gc4c75lg7b4lsar.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_42 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_22, [None, None, %unsqueeze_38, %convert_element_type_501]), kwargs = {})
#   %convert_element_type_544 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_42, torch.bfloat16), kwargs = {})
#   %add_455 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2994, %convert_element_type_544), kwargs = {})
#   %convolution_backward_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_2995, %add_455, %expand_86, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 56) % 56)
    x0 = (xindex % 56)
    x2 = ((xindex // 3136) % 64)
    x3 = xindex // 200704
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.32142857142857145
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 64*tmp9 + 1152*tmp5 + 20736*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xg/cxggxpcgjpgmvl3uquy6452zxapvgbz5szsyvs2gzfrb3uldmt3o.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_545 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1194, torch.float32), kwargs = {})
triton_poi_fused__to_copy_75 = async_compile.triton('triton_poi_fused__to_copy_75', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_75', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 163840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_75(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3x/c3xlxdbvvytnbdy2g7fkwsq2ztwnx3beme3fc422y2tcijrrgtc5.py
# Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   bn1 => full_default
# Graph fragment:
#   %full_default : [num_users=16] = call_function[target=torch.ops.aten.full.default](args = ([64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_76 = async_compile.triton('triton_poi_fused_zeros_76', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_76', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_76(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yb/cybvodfic4ly4se52tpvuudp73olmb47czlmgbfyqrwmnzik35pv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_19, clone_default_20, clone_default_22, clone_default_23
# Graph fragment:
#   %clone_default_22 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_44,), kwargs = {})
#   %clone_default_23 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_46,), kwargs = {})
#   %clone_default_19 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_38,), kwargs = {})
#   %clone_default_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_40,), kwargs = {})
triton_poi_fused_77 = async_compile.triton('triton_poi_fused_77', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_77', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_77(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/ie/cienymejzwyxlrf6lz6s3goa7yuhdxcpl4yhigzaf6sbhm62gzne.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_45, triton_kernel_wrapper_mutation_46
# Graph fragment:
#   %triton_kernel_wrapper_mutation_46 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 609, constant_args_idx: 837, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3016, DY: %view_3014, DBETA: %as_strided_default_45, DGAMMA: %as_strided_default_47, M: 6422528, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_45 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 610, constant_args_idx: 838, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3016, DY: %view_3014, INVSTD: %rsqrt_9, GAMMA: %primals_58, DBETA: %as_strided_default_45, DGAMMA: %as_strided_default_47, DX: %permute_200, M: 6422528, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_78 = async_compile.triton('triton_poi_fused_78', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_78', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_78(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_22 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_22', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_23 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_23', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sp/cspuixnzgljdksddklzre3hjnjbm2n4mc7pvr6rsuzttagftsm77.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_555 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1206, torch.float32), kwargs = {})
triton_poi_fused__to_copy_79 = async_compile.triton('triton_poi_fused__to_copy_79', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_79', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 368640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_79(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/l3/cl3mqavfpyh4llrhli7sur2yuqtkvj23mswzt2t4gjdsqftyowds.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_44 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_18, [None, None, %unsqueeze_38, %convert_element_type_501]), kwargs = {})
#   %convert_element_type_564 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_44, torch.bfloat16), kwargs = {})
#   %add_465 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3072, %convert_element_type_564), kwargs = {})
#   %convolution_backward_89 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_3073, %add_465, %expand_90, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_80 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_80', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_80', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_80(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 56) % 56)
    x0 = (xindex % 56)
    x2 = ((xindex // 3136) % 256)
    x3 = xindex // 802816
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.32142857142857145
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 256*tmp9 + 4608*tmp5 + 82944*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/dq/cdqkb7yzsqyoyzbmff5dzd7mfgfs4o3reb6o54xmsx3h7o2dyqnl.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_18, clone_default_9
# Graph fragment:
#   %clone_default_18 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_36,), kwargs = {})
#   %clone_default_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_18,), kwargs = {})
triton_poi_fused_81 = async_compile.triton('triton_poi_fused_81', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_81', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8220835840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_81(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/l4/cl4axgekeul7lzsbut4b7z4p6yt3g3lbcwmwrbyitg66mwohjtrj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_450 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1143, %getitem_1178), kwargs = {})
#   %mul_578 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_450, %view_2960), kwargs = {})
#   %add_466 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_578, %getitem_1214), kwargs = {})
#   %mul_593 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_466, %view_3077), kwargs = {})
triton_poi_fused_add_mul_82 = async_compile.triton('triton_poi_fused_add_mul_82', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'in_ptr3': '*bf16', 'in_ptr4': '*i8', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_82', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9865003008, 'x': 6576668672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_82(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 256*x2 + 802816*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + 256*x2 + 802816*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/cq/ccqkj5zk7own3bkeor4vjcr75ueopm67bdjesz7ja4qh7ydtrw64.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_6, clone_default_7, clone_default_8
# Graph fragment:
#   %clone_default_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_14,), kwargs = {})
#   %clone_default_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_16,), kwargs = {})
#   %clone_default_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_12,), kwargs = {})
triton_poi_fused_83 = async_compile.triton('triton_poi_fused_83', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_83', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_83(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/4r/c4rbh3o5nxqkh2hwhyf4qidyacus5yryfrqggxnnfnlfgksdcv5p.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_16, triton_kernel_wrapper_mutation_17, triton_kernel_wrapper_mutation_20, triton_kernel_wrapper_mutation_21
# Graph fragment:
#   %triton_kernel_wrapper_mutation_21 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 629, constant_args_idx: 862, grid: [(256, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3211, DY: %view_3209, DBETA: %as_strided_default_15, DGAMMA: %as_strided_default_17, M: 6422528, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_20 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 630, constant_args_idx: 863, grid: [(256, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3211, DY: %view_3209, INVSTD: %rsqrt_4, GAMMA: %primals_28, DBETA: %as_strided_default_15, DGAMMA: %as_strided_default_17, DX: %permute_205, M: 6422528, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_17 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 633, constant_args_idx: 866, grid: [(256, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3246, DY: %view_3209, DBETA: %full_default_18, DGAMMA: %as_strided_default_13, M: 6422528, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_16 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 634, constant_args_idx: 867, grid: [(256, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3246, DY: %view_3209, INVSTD: %rsqrt_3, GAMMA: %primals_22, DBETA: %full_default_18, DGAMMA: %as_strided_default_13, DX: %permute_206, M: 6422528, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_84 = async_compile.triton('triton_poi_fused_84', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_84', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 3288334336, 'x': 29595009024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_84(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 256*x2 + 802816*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp5, xmask & ymask)
    tl.store(out_ptr2 + (x2 + 3136*y3), tmp5, xmask & ymask)
    tl.store(out_ptr3 + (x2 + 3136*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/oz/cozcm3dplijfpl2j72e43sjhe3zlhlaqzqn7ugyejnfkatnmnpvt.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_48 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_4, [None, None, %unsqueeze_38, %convert_element_type_501]), kwargs = {})
#   %convert_element_type_604 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_48, torch.bfloat16), kwargs = {})
#   %add_487 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3228, %convert_element_type_604), kwargs = {})
#   %convolution_backward_97 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_3229, %add_487, %expand_98, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
#   %add_502 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3341, %convert_element_type_604), kwargs = {})
#   %convolution_backward_103 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_3342, %add_502, %expand_104, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_85 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_85', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_85', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_85(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 56) % 56)
    x0 = (xindex % 56)
    x2 = ((xindex // 3136) % 64)
    x3 = xindex // 200704
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.32142857142857145
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 64*tmp9 + 1152*tmp5 + 20736*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tmp14 = tmp13 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
    tl.store(out_ptr1 + (x4), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qb/cqbxebcdp6ws5fqnecrc4hncd34seh24s7qbijifaydg5vnsaqy3.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default, clone_default_1, clone_default_2, clone_default_3, clone_default_4
# Graph fragment:
#   %clone_default_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_6,), kwargs = {})
#   %clone_default_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_8,), kwargs = {})
#   %clone_default_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_2,), kwargs = {})
#   %clone_default_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_4,), kwargs = {})
#   %clone_default : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default,), kwargs = {})
triton_poi_fused_86 = async_compile.triton('triton_poi_fused_86', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_86', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_86(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wg/cwg25pkpnskjesiqzxpwud344c5s3izwiqrd6ozvx75odxfh3d7k.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_635 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1301, torch.float32), kwargs = {})
triton_poi_fused__to_copy_87 = async_compile.triton('triton_poi_fused__to_copy_87', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_87', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 40960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_87(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/od/codxhbhajqemy5uoq52rbez5wcby5n5nlufoarn22brrxewa2h2o.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_503 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1262, %getitem_1297), kwargs = {})
triton_poi_fused_add_88 = async_compile.triton('triton_poi_fused_add_88', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_88', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_88(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sc/csc4g7hbwfppofvn2ewyorwd2kadwbuzg5ubin57nwfx3ldjejkh.py
# Topologically Sorted Source Nodes: [maxpool], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
# Source node to ATen node mapping:
#   maxpool => _low_memory_max_pool_offsets_to_indices
# Graph fragment:
#   %add_503 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1262, %getitem_1297), kwargs = {})
#   %_low_memory_max_pool_offsets_to_indices : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_offsets_to_indices.default](args = (%getitem_14, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]), kwargs = {})
#   %max_pool2d_with_indices_backward : [num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%add_503, %getitem_10, [3, 3], [2, 2], [1, 1], [1, 1], False, %_low_memory_max_pool_offsets_to_indices), kwargs = {})
triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_89 = async_compile.triton('triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_89', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 33554432, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_89', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_89(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25690112
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 112)
    y1 = ((yindex // 112) % 112)
    y2 = yindex // 12544
    y4 = (yindex % 12544)
    tmp0 = tl.load(in_ptr0 + (x3 + 64*((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 3584*((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))))) + 200704*y2), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x3 + 64*((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 3584*((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))))) + 200704*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (x3 + 64*((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 3584*((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))))) + 200704*y2), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3 + 64*((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 3584*((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))))) + 200704*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp30 = tl.load(in_ptr0 + (x3 + 64*((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 3584*((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))))) + 200704*y2), xmask & ymask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (x3 + 64*((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 3584*((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))))) + 200704*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp46 = tl.load(in_ptr0 + (x3 + 64*((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 3584*((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))))) + 200704*y2), xmask & ymask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (x3 + 64*((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 3584*((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))))) + 200704*y2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([YBLOCK, XBLOCK], 9, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 9)) | ~(xmask & ymask), "index out of bounds: 0 <= tmp4 < 9")
    tmp7 = (-113) + tmp4 + 2*((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 109*(tmp4 // 3) + 224*((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0))))))
    tmp8 = y4
    tmp9 = tmp7 == tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tmp13 = tmp12 + tmp1
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert(((0 <= tmp15) & (tmp15 < 9)) | ~(xmask & ymask), "index out of bounds: 0 <= tmp15 < 9")
    tmp18 = (-113) + tmp15 + 2*((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 109*(tmp15 // 3) + 224*((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0))))))
    tmp19 = tmp18 == tmp8
    tmp20 = ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))
    tmp21 = ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))
    tmp22 = tmp20 < tmp21
    tmp23 = 1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))
    tmp24 = ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))
    tmp25 = tmp23 < tmp24
    tmp26 = tmp22 & tmp25
    tmp27 = tmp26 & tmp19
    tmp28 = tmp11 + tmp17
    tmp29 = tl.where(tmp27, tmp28, tmp11)
    tmp31 = tmp30 + tmp1
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.device_assert(((0 <= tmp33) & (tmp33 < 9)) | ~(xmask & ymask), "index out of bounds: 0 <= tmp33 < 9")
    tmp36 = (-113) + tmp33 + 2*((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 109*(tmp33 // 3) + 224*((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0))))))
    tmp37 = tmp36 == tmp8
    tmp38 = 1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))
    tmp39 = tmp38 < tmp21
    tmp40 = ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))
    tmp41 = tmp40 < tmp24
    tmp42 = tmp39 & tmp41
    tmp43 = tmp42 & tmp37
    tmp44 = tmp29 + tmp35
    tmp45 = tl.where(tmp43, tmp44, tmp29)
    tmp47 = tmp46 + tmp1
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert(((0 <= tmp49) & (tmp49 < 9)) | ~(xmask & ymask), "index out of bounds: 0 <= tmp49 < 9")
    tmp52 = (-113) + tmp49 + 2*((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y0) // 2))) + (1 + ((1 + y0) // 2)) * ((1 + ((1 + y0) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y0 // 2)) + (y0 // 2) * ((y0 // 2) > (0)))))) + 109*(tmp49 // 3) + 224*((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + y1) // 2))) + (1 + ((1 + y1) // 2)) * ((1 + ((1 + y1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (y1 // 2)) + (y1 // 2) * ((y1 // 2) > (0))))))
    tmp53 = tmp52 == tmp8
    tmp54 = tmp39 & tmp25
    tmp55 = tmp54 & tmp53
    tmp56 = tmp45 + tmp51
    tmp57 = tl.where(tmp55, tmp56, tmp45)
    tl.store(out_ptr0 + (y4 + 12544*x3 + 802816*y2), tmp57, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4y/c4ya6skzahbdvvimigrwc4qnojdfnt2oslbugophhtfp74artnsc.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_1, triton_kernel_wrapper_mutation_2
# Graph fragment:
#   %triton_kernel_wrapper_mutation_2 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 645, constant_args_idx: 881, grid: [(64, 25088, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3363, DY: %view_3361, DBETA: %full_default, DGAMMA: %as_strided_default_1, M: 25690112, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_1 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 646, constant_args_idx: 882, grid: [(64, 25088, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3363, DY: %view_3361, INVSTD: %rsqrt, GAMMA: %primals_4, DBETA: %full_default, DGAMMA: %as_strided_default_1, DX: %permute_209, M: 25690112, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
triton_poi_fused_90 = async_compile.triton('triton_poi_fused_90', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2147483648}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_90', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16441671680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_90(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1644167168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)
    tl.store(out_ptr1 + (x0), tmp3, None)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:508
_bn_bwd_reduce_kernel_24 = async_compile.triton('_bn_bwd_reduce_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_reduce_kernel_24', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT6/ops.py:538
_bn_bwd_dx_kernel_25 = async_compile.triton('_bn_bwd_dx_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 8, 'num_stages': 5}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_bn_bwd_dx_kernel_25', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X_hat': '*bf16', 'DY': '*bf16', 'INVSTD': '*fp32', 'GAMMA': '*fp32', 'DBETA': '*fp32', 'DGAMMA': '*fp32', 'DX': '*bf16', 'M': 'i32', 'HW': 'i32', 'stride_n': 'i32', 'stride_c': 'i32', 'BLOCK_M': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'BLOCK_M': 1024}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, INVSTD, GAMMA,
    DBETA, DGAMMA,
    DX,
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

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    invstd = tl.load(INVSTD + c).to(tl.float32)
    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bl/cbl4jlk5ngvifxbgqna2f24y6d65e7vsswvfxwvlezb2uow77nak.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %_unsafe_index_52 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convert_element_type_2, [None, None, %unsqueeze_52, %convert_element_type_641]), kwargs = {})
#   %convert_element_type_644 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%_unsafe_index_52, torch.bfloat16), kwargs = {})
#   %add_508 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3380, %convert_element_type_644), kwargs = {})
#   %convolution_backward_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%view_3381, %add_508, %expand_105, None, [2, 2], [3, 3], [1, 1], False, [0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_91 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_91', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp8e4nv', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_91', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_91(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 308281344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x1 = ((xindex // 224) % 224)
    x0 = (xindex % 224)
    x2 = ((xindex // 50176) % 3)
    x3 = xindex // 150528
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp1 = x1
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.33035714285714285
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = x0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.load(in_ptr1 + (x2 + 3*tmp9 + 222*tmp5 + 16428*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp0 + tmp11
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mi/cmiah7fcjxizt3g3s52omi4hwyzytaklleyvfhukfyodem3gjygl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_645 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1310, torch.float32), kwargs = {})
triton_poi_fused__to_copy_92 = async_compile.triton('triton_poi_fused__to_copy_92', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_92', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 94080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_92(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, convert_element_type_2, getitem, getitem_1, getitem_2, rsqrt, getitem_7, getitem_8, getitem_9, getitem_10, getitem_12, getitem_14, convert_element_type_3, convert_element_type_4, getitem_15, getitem_16, getitem_17, rsqrt_1, getitem_22, getitem_23, getitem_24, getitem_27, convert_element_type_5, convert_element_type_6, getitem_28, getitem_29, getitem_30, rsqrt_2, getitem_35, getitem_36, getitem_37, getitem_40, convert_element_type_7, convert_element_type_8, getitem_41, getitem_42, getitem_43, rsqrt_3, getitem_48, getitem_49, getitem_50, convert_element_type_9, getitem_51, getitem_52, getitem_53, rsqrt_4, getitem_58, getitem_59, getitem_60, getitem_63, convert_element_type_11, convert_element_type_12, getitem_64, getitem_65, getitem_66, rsqrt_5, getitem_71, getitem_72, getitem_73, getitem_76, convert_element_type_13, convert_element_type_14, getitem_77, getitem_78, getitem_79, rsqrt_6, getitem_84, getitem_85, getitem_86, getitem_89, convert_element_type_15, convert_element_type_16, getitem_90, getitem_91, getitem_92, rsqrt_7, getitem_97, getitem_98, getitem_99, getitem_102, convert_element_type_17, convert_element_type_18, getitem_103, getitem_104, getitem_105, rsqrt_8, getitem_110, getitem_111, getitem_112, getitem_115, convert_element_type_19, convert_element_type_20, getitem_116, getitem_117, getitem_118, rsqrt_9, getitem_123, getitem_124, getitem_125, getitem_128, convert_element_type_21, convert_element_type_22, getitem_129, getitem_130, getitem_131, rsqrt_10, getitem_136, getitem_137, getitem_138, getitem_141, convert_element_type_23, convert_element_type_24, getitem_142, getitem_143, getitem_144, rsqrt_11, getitem_149, getitem_150, getitem_151, getitem_154, convert_element_type_25, convert_element_type_26, getitem_155, getitem_156, getitem_157, rsqrt_12, getitem_162, getitem_163, getitem_164, getitem_167, convert_element_type_27, convert_element_type_28, getitem_168, getitem_169, getitem_170, rsqrt_13, getitem_175, getitem_176, getitem_177, convert_element_type_29, getitem_178, getitem_179, getitem_180, rsqrt_14, getitem_185, getitem_186, getitem_187, getitem_190, convert_element_type_31, convert_element_type_32, getitem_191, getitem_192, getitem_193, rsqrt_15, getitem_198, getitem_199, getitem_200, getitem_203, convert_element_type_33, convert_element_type_34, getitem_204, getitem_205, getitem_206, rsqrt_16, getitem_211, getitem_212, getitem_213, getitem_216, convert_element_type_35, convert_element_type_36, getitem_217, getitem_218, getitem_219, rsqrt_17, getitem_224, getitem_225, getitem_226, getitem_229, convert_element_type_37, convert_element_type_38, getitem_230, getitem_231, getitem_232, rsqrt_18, getitem_237, getitem_238, getitem_239, getitem_242, convert_element_type_39, convert_element_type_40, getitem_243, getitem_244, getitem_245, rsqrt_19, getitem_250, getitem_251, getitem_252, getitem_255, convert_element_type_41, convert_element_type_42, getitem_256, getitem_257, getitem_258, rsqrt_20, getitem_263, getitem_264, getitem_265, getitem_268, convert_element_type_43, convert_element_type_44, getitem_269, getitem_270, getitem_271, rsqrt_21, getitem_276, getitem_277, getitem_278, getitem_281, convert_element_type_45, convert_element_type_46, getitem_282, getitem_283, getitem_284, rsqrt_22, getitem_289, getitem_290, getitem_291, getitem_294, convert_element_type_47, convert_element_type_48, getitem_295, getitem_296, getitem_297, rsqrt_23, getitem_302, getitem_303, getitem_304, getitem_307, convert_element_type_49, convert_element_type_50, getitem_308, getitem_309, getitem_310, rsqrt_24, getitem_315, getitem_316, getitem_317, getitem_320, convert_element_type_51, convert_element_type_52, getitem_321, getitem_322, getitem_323, rsqrt_25, getitem_328, getitem_329, getitem_330, getitem_333, convert_element_type_53, convert_element_type_54, getitem_334, getitem_335, getitem_336, rsqrt_26, getitem_341, getitem_342, getitem_343, convert_element_type_55, getitem_344, getitem_345, getitem_346, rsqrt_27, getitem_351, getitem_352, getitem_353, getitem_356, convert_element_type_57, convert_element_type_58, getitem_357, getitem_358, getitem_359, rsqrt_28, getitem_364, getitem_365, getitem_366, getitem_369, convert_element_type_59, convert_element_type_60, getitem_370, getitem_371, getitem_372, rsqrt_29, getitem_377, getitem_378, getitem_379, getitem_382, convert_element_type_61, convert_element_type_62, getitem_383, getitem_384, getitem_385, rsqrt_30, getitem_390, getitem_391, getitem_392, getitem_395, convert_element_type_63, convert_element_type_64, getitem_396, getitem_397, getitem_398, rsqrt_31, getitem_403, getitem_404, getitem_405, getitem_408, convert_element_type_65, convert_element_type_66, getitem_409, getitem_410, getitem_411, rsqrt_32, getitem_416, getitem_417, getitem_418, getitem_421, convert_element_type_67, convert_element_type_68, getitem_422, getitem_423, getitem_424, rsqrt_33, getitem_429, getitem_430, getitem_431, getitem_434, convert_element_type_69, convert_element_type_70, getitem_435, getitem_436, getitem_437, rsqrt_34, getitem_442, getitem_443, getitem_444, getitem_447, convert_element_type_71, convert_element_type_72, getitem_448, getitem_449, getitem_450, rsqrt_35, getitem_455, getitem_456, getitem_457, getitem_460, convert_element_type_73, convert_element_type_74, getitem_461, getitem_462, getitem_463, rsqrt_36, getitem_468, getitem_469, getitem_470, getitem_473, convert_element_type_75, convert_element_type_76, getitem_474, getitem_475, getitem_476, rsqrt_37, getitem_481, getitem_482, getitem_483, getitem_486, convert_element_type_77, convert_element_type_78, getitem_487, getitem_488, getitem_489, rsqrt_38, getitem_494, getitem_495, getitem_496, getitem_499, convert_element_type_79, convert_element_type_80, getitem_500, getitem_501, getitem_502, rsqrt_39, getitem_507, getitem_508, getitem_509, getitem_512, convert_element_type_81, convert_element_type_82, getitem_513, getitem_514, getitem_515, rsqrt_40, getitem_520, getitem_521, getitem_522, getitem_525, convert_element_type_83, convert_element_type_84, getitem_526, getitem_527, getitem_528, rsqrt_41, getitem_533, getitem_534, getitem_535, getitem_538, convert_element_type_85, convert_element_type_86, getitem_539, getitem_540, getitem_541, rsqrt_42, getitem_546, getitem_547, getitem_548, getitem_551, convert_element_type_87, convert_element_type_88, getitem_552, getitem_553, getitem_554, rsqrt_43, getitem_559, getitem_560, getitem_561, getitem_564, convert_element_type_89, convert_element_type_90, getitem_565, getitem_566, getitem_567, rsqrt_44, getitem_572, getitem_573, getitem_574, getitem_577, convert_element_type_91, convert_element_type_92, getitem_578, getitem_579, getitem_580, rsqrt_45, getitem_585, getitem_586, getitem_587, convert_element_type_93, getitem_588, getitem_589, getitem_590, rsqrt_46, getitem_595, getitem_596, getitem_597, getitem_600, convert_element_type_95, convert_element_type_96, getitem_601, getitem_602, getitem_603, rsqrt_47, getitem_608, getitem_609, getitem_610, getitem_613, convert_element_type_97, convert_element_type_98, getitem_614, getitem_615, getitem_616, rsqrt_48, getitem_621, getitem_622, getitem_623, getitem_626, convert_element_type_99, convert_element_type_100, getitem_627, getitem_628, getitem_629, rsqrt_49, getitem_634, getitem_635, getitem_636, getitem_639, convert_element_type_101, convert_element_type_102, getitem_640, getitem_641, getitem_642, rsqrt_50, getitem_647, getitem_648, getitem_649, getitem_652, convert_element_type_103, convert_element_type_104, getitem_653, getitem_654, getitem_655, rsqrt_51, getitem_660, getitem_661, getitem_662, getitem_665, convert_element_type_105, convert_element_type_106, getitem_666, getitem_667, getitem_668, rsqrt_52, getitem_673, getitem_674, getitem_675, getitem_678, getitem_679, getitem_680, getitem_681, convert_element_type_107, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_52, (64, ), (1, ))
    assert_size_stride(primals_58, (64, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_118, (128, ), (1, ))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_136, (128, ), (1, ))
    assert_size_stride(primals_142, (512, ), (1, ))
    assert_size_stride(primals_148, (256, ), (1, ))
    assert_size_stride(primals_154, (256, ), (1, ))
    assert_size_stride(primals_160, (1024, ), (1, ))
    assert_size_stride(primals_166, (1024, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_178, (256, ), (1, ))
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_190, (256, ), (1, ))
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_202, (1024, ), (1, ))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_220, (1024, ), (1, ))
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_232, (256, ), (1, ))
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_256, (1024, ), (1, ))
    assert_size_stride(primals_262, (512, ), (1, ))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_274, (2048, ), (1, ))
    assert_size_stride(primals_280, (2048, ), (1, ))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_298, (2048, ), (1, ))
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_316, (2048, ), (1, ))
    assert_size_stride(convert_element_type_2, (2048, 3, 74, 74), (16428, 1, 222, 3))
    assert_size_stride(getitem, (602112, 32), (32, 1))
    assert_size_stride(getitem_1, (602112, ), (1, ))
    assert_size_stride(getitem_2, (602112, ), (1, ))
    assert_size_stride(rsqrt, (64, ), (1, ))
    assert_size_stride(getitem_7, (3211264, 32), (32, 1))
    assert_size_stride(getitem_8, (3211264, ), (1, ))
    assert_size_stride(getitem_9, (3211264, ), (1, ))
    assert_size_stride(getitem_10, (2048, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(getitem_12, (3211264, 16), (16, 1))
    assert_size_stride(getitem_14, (2048, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_3, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convert_element_type_4, (2048, 64, 18, 18), (20736, 1, 1152, 64))
    assert_size_stride(getitem_15, (802816, 32), (32, 1))
    assert_size_stride(getitem_16, (802816, ), (1, ))
    assert_size_stride(getitem_17, (802816, ), (1, ))
    assert_size_stride(rsqrt_1, (64, ), (1, ))
    assert_size_stride(getitem_22, (802816, 32), (32, 1))
    assert_size_stride(getitem_23, (802816, ), (1, ))
    assert_size_stride(getitem_24, (802816, ), (1, ))
    assert_size_stride(getitem_27, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_5, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convert_element_type_6, (2048, 64, 18, 18), (20736, 1, 1152, 64))
    assert_size_stride(getitem_28, (802816, 32), (32, 1))
    assert_size_stride(getitem_29, (802816, ), (1, ))
    assert_size_stride(getitem_30, (802816, ), (1, ))
    assert_size_stride(rsqrt_2, (64, ), (1, ))
    assert_size_stride(getitem_35, (802816, 32), (32, 1))
    assert_size_stride(getitem_36, (802816, ), (1, ))
    assert_size_stride(getitem_37, (802816, ), (1, ))
    assert_size_stride(getitem_40, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_7, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convert_element_type_8, (2048, 64, 18, 18), (20736, 1, 1152, 64))
    assert_size_stride(getitem_41, (802816, 32), (32, 1))
    assert_size_stride(getitem_42, (802816, ), (1, ))
    assert_size_stride(getitem_43, (802816, ), (1, ))
    assert_size_stride(rsqrt_3, (256, ), (1, ))
    assert_size_stride(getitem_48, (3211264, 32), (32, 1))
    assert_size_stride(getitem_49, (3211264, ), (1, ))
    assert_size_stride(getitem_50, (3211264, ), (1, ))
    assert_size_stride(convert_element_type_9, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_51, (802816, 32), (32, 1))
    assert_size_stride(getitem_52, (802816, ), (1, ))
    assert_size_stride(getitem_53, (802816, ), (1, ))
    assert_size_stride(rsqrt_4, (256, ), (1, ))
    assert_size_stride(getitem_58, (3211264, 32), (32, 1))
    assert_size_stride(getitem_59, (3211264, ), (1, ))
    assert_size_stride(getitem_60, (3211264, ), (1, ))
    assert_size_stride(getitem_63, (3211264, 16), (16, 1))
    assert_size_stride(convert_element_type_11, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convert_element_type_12, (2048, 256, 18, 18), (82944, 1, 4608, 256))
    assert_size_stride(getitem_64, (3211264, 32), (32, 1))
    assert_size_stride(getitem_65, (3211264, ), (1, ))
    assert_size_stride(getitem_66, (3211264, ), (1, ))
    assert_size_stride(rsqrt_5, (64, ), (1, ))
    assert_size_stride(getitem_71, (802816, 32), (32, 1))
    assert_size_stride(getitem_72, (802816, ), (1, ))
    assert_size_stride(getitem_73, (802816, ), (1, ))
    assert_size_stride(getitem_76, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_13, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convert_element_type_14, (2048, 64, 18, 18), (20736, 1, 1152, 64))
    assert_size_stride(getitem_77, (802816, 32), (32, 1))
    assert_size_stride(getitem_78, (802816, ), (1, ))
    assert_size_stride(getitem_79, (802816, ), (1, ))
    assert_size_stride(rsqrt_6, (64, ), (1, ))
    assert_size_stride(getitem_84, (802816, 32), (32, 1))
    assert_size_stride(getitem_85, (802816, ), (1, ))
    assert_size_stride(getitem_86, (802816, ), (1, ))
    assert_size_stride(getitem_89, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_15, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convert_element_type_16, (2048, 64, 18, 18), (20736, 1, 1152, 64))
    assert_size_stride(getitem_90, (802816, 32), (32, 1))
    assert_size_stride(getitem_91, (802816, ), (1, ))
    assert_size_stride(getitem_92, (802816, ), (1, ))
    assert_size_stride(rsqrt_7, (256, ), (1, ))
    assert_size_stride(getitem_97, (3211264, 32), (32, 1))
    assert_size_stride(getitem_98, (3211264, ), (1, ))
    assert_size_stride(getitem_99, (3211264, ), (1, ))
    assert_size_stride(getitem_102, (3211264, 16), (16, 1))
    assert_size_stride(convert_element_type_17, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convert_element_type_18, (2048, 256, 18, 18), (82944, 1, 4608, 256))
    assert_size_stride(getitem_103, (3211264, 32), (32, 1))
    assert_size_stride(getitem_104, (3211264, ), (1, ))
    assert_size_stride(getitem_105, (3211264, ), (1, ))
    assert_size_stride(rsqrt_8, (64, ), (1, ))
    assert_size_stride(getitem_110, (802816, 32), (32, 1))
    assert_size_stride(getitem_111, (802816, ), (1, ))
    assert_size_stride(getitem_112, (802816, ), (1, ))
    assert_size_stride(getitem_115, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_19, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convert_element_type_20, (2048, 64, 18, 18), (20736, 1, 1152, 64))
    assert_size_stride(getitem_116, (802816, 32), (32, 1))
    assert_size_stride(getitem_117, (802816, ), (1, ))
    assert_size_stride(getitem_118, (802816, ), (1, ))
    assert_size_stride(rsqrt_9, (64, ), (1, ))
    assert_size_stride(getitem_123, (802816, 32), (32, 1))
    assert_size_stride(getitem_124, (802816, ), (1, ))
    assert_size_stride(getitem_125, (802816, ), (1, ))
    assert_size_stride(getitem_128, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_21, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convert_element_type_22, (2048, 64, 18, 18), (20736, 1, 1152, 64))
    assert_size_stride(getitem_129, (802816, 32), (32, 1))
    assert_size_stride(getitem_130, (802816, ), (1, ))
    assert_size_stride(getitem_131, (802816, ), (1, ))
    assert_size_stride(rsqrt_10, (256, ), (1, ))
    assert_size_stride(getitem_136, (3211264, 32), (32, 1))
    assert_size_stride(getitem_137, (3211264, ), (1, ))
    assert_size_stride(getitem_138, (3211264, ), (1, ))
    assert_size_stride(getitem_141, (3211264, 16), (16, 1))
    assert_size_stride(convert_element_type_23, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convert_element_type_24, (2048, 256, 18, 18), (82944, 1, 4608, 256))
    assert_size_stride(getitem_142, (3211264, 32), (32, 1))
    assert_size_stride(getitem_143, (3211264, ), (1, ))
    assert_size_stride(getitem_144, (3211264, ), (1, ))
    assert_size_stride(rsqrt_11, (128, ), (1, ))
    assert_size_stride(getitem_149, (1605632, 32), (32, 1))
    assert_size_stride(getitem_150, (1605632, ), (1, ))
    assert_size_stride(getitem_151, (1605632, ), (1, ))
    assert_size_stride(getitem_154, (1605632, 16), (16, 1))
    assert_size_stride(convert_element_type_25, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convert_element_type_26, (2048, 128, 18, 18), (41472, 1, 2304, 128))
    assert_size_stride(getitem_155, (1605632, 32), (32, 1))
    assert_size_stride(getitem_156, (1605632, ), (1, ))
    assert_size_stride(getitem_157, (1605632, ), (1, ))
    assert_size_stride(rsqrt_12, (128, ), (1, ))
    assert_size_stride(getitem_162, (401408, 32), (32, 1))
    assert_size_stride(getitem_163, (401408, ), (1, ))
    assert_size_stride(getitem_164, (401408, ), (1, ))
    assert_size_stride(getitem_167, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_27, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convert_element_type_28, (2048, 128, 9, 9), (10368, 1, 1152, 128))
    assert_size_stride(getitem_168, (401408, 32), (32, 1))
    assert_size_stride(getitem_169, (401408, ), (1, ))
    assert_size_stride(getitem_170, (401408, ), (1, ))
    assert_size_stride(rsqrt_13, (512, ), (1, ))
    assert_size_stride(getitem_175, (1605632, 32), (32, 1))
    assert_size_stride(getitem_176, (1605632, ), (1, ))
    assert_size_stride(getitem_177, (1605632, ), (1, ))
    assert_size_stride(convert_element_type_29, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_178, (3211264, 32), (32, 1))
    assert_size_stride(getitem_179, (3211264, ), (1, ))
    assert_size_stride(getitem_180, (3211264, ), (1, ))
    assert_size_stride(rsqrt_14, (512, ), (1, ))
    assert_size_stride(getitem_185, (1605632, 32), (32, 1))
    assert_size_stride(getitem_186, (1605632, ), (1, ))
    assert_size_stride(getitem_187, (1605632, ), (1, ))
    assert_size_stride(getitem_190, (1605632, 16), (16, 1))
    assert_size_stride(convert_element_type_31, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convert_element_type_32, (2048, 512, 9, 9), (41472, 1, 4608, 512))
    assert_size_stride(getitem_191, (1605632, 32), (32, 1))
    assert_size_stride(getitem_192, (1605632, ), (1, ))
    assert_size_stride(getitem_193, (1605632, ), (1, ))
    assert_size_stride(rsqrt_15, (128, ), (1, ))
    assert_size_stride(getitem_198, (401408, 32), (32, 1))
    assert_size_stride(getitem_199, (401408, ), (1, ))
    assert_size_stride(getitem_200, (401408, ), (1, ))
    assert_size_stride(getitem_203, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_33, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convert_element_type_34, (2048, 128, 9, 9), (10368, 1, 1152, 128))
    assert_size_stride(getitem_204, (401408, 32), (32, 1))
    assert_size_stride(getitem_205, (401408, ), (1, ))
    assert_size_stride(getitem_206, (401408, ), (1, ))
    assert_size_stride(rsqrt_16, (128, ), (1, ))
    assert_size_stride(getitem_211, (401408, 32), (32, 1))
    assert_size_stride(getitem_212, (401408, ), (1, ))
    assert_size_stride(getitem_213, (401408, ), (1, ))
    assert_size_stride(getitem_216, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_35, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convert_element_type_36, (2048, 128, 9, 9), (10368, 1, 1152, 128))
    assert_size_stride(getitem_217, (401408, 32), (32, 1))
    assert_size_stride(getitem_218, (401408, ), (1, ))
    assert_size_stride(getitem_219, (401408, ), (1, ))
    assert_size_stride(rsqrt_17, (512, ), (1, ))
    assert_size_stride(getitem_224, (1605632, 32), (32, 1))
    assert_size_stride(getitem_225, (1605632, ), (1, ))
    assert_size_stride(getitem_226, (1605632, ), (1, ))
    assert_size_stride(getitem_229, (1605632, 16), (16, 1))
    assert_size_stride(convert_element_type_37, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convert_element_type_38, (2048, 512, 9, 9), (41472, 1, 4608, 512))
    assert_size_stride(getitem_230, (1605632, 32), (32, 1))
    assert_size_stride(getitem_231, (1605632, ), (1, ))
    assert_size_stride(getitem_232, (1605632, ), (1, ))
    assert_size_stride(rsqrt_18, (128, ), (1, ))
    assert_size_stride(getitem_237, (401408, 32), (32, 1))
    assert_size_stride(getitem_238, (401408, ), (1, ))
    assert_size_stride(getitem_239, (401408, ), (1, ))
    assert_size_stride(getitem_242, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_39, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convert_element_type_40, (2048, 128, 9, 9), (10368, 1, 1152, 128))
    assert_size_stride(getitem_243, (401408, 32), (32, 1))
    assert_size_stride(getitem_244, (401408, ), (1, ))
    assert_size_stride(getitem_245, (401408, ), (1, ))
    assert_size_stride(rsqrt_19, (128, ), (1, ))
    assert_size_stride(getitem_250, (401408, 32), (32, 1))
    assert_size_stride(getitem_251, (401408, ), (1, ))
    assert_size_stride(getitem_252, (401408, ), (1, ))
    assert_size_stride(getitem_255, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_41, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convert_element_type_42, (2048, 128, 9, 9), (10368, 1, 1152, 128))
    assert_size_stride(getitem_256, (401408, 32), (32, 1))
    assert_size_stride(getitem_257, (401408, ), (1, ))
    assert_size_stride(getitem_258, (401408, ), (1, ))
    assert_size_stride(rsqrt_20, (512, ), (1, ))
    assert_size_stride(getitem_263, (1605632, 32), (32, 1))
    assert_size_stride(getitem_264, (1605632, ), (1, ))
    assert_size_stride(getitem_265, (1605632, ), (1, ))
    assert_size_stride(getitem_268, (1605632, 16), (16, 1))
    assert_size_stride(convert_element_type_43, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convert_element_type_44, (2048, 512, 9, 9), (41472, 1, 4608, 512))
    assert_size_stride(getitem_269, (1605632, 32), (32, 1))
    assert_size_stride(getitem_270, (1605632, ), (1, ))
    assert_size_stride(getitem_271, (1605632, ), (1, ))
    assert_size_stride(rsqrt_21, (128, ), (1, ))
    assert_size_stride(getitem_276, (401408, 32), (32, 1))
    assert_size_stride(getitem_277, (401408, ), (1, ))
    assert_size_stride(getitem_278, (401408, ), (1, ))
    assert_size_stride(getitem_281, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_45, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convert_element_type_46, (2048, 128, 9, 9), (10368, 1, 1152, 128))
    assert_size_stride(getitem_282, (401408, 32), (32, 1))
    assert_size_stride(getitem_283, (401408, ), (1, ))
    assert_size_stride(getitem_284, (401408, ), (1, ))
    assert_size_stride(rsqrt_22, (128, ), (1, ))
    assert_size_stride(getitem_289, (401408, 32), (32, 1))
    assert_size_stride(getitem_290, (401408, ), (1, ))
    assert_size_stride(getitem_291, (401408, ), (1, ))
    assert_size_stride(getitem_294, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_47, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convert_element_type_48, (2048, 128, 9, 9), (10368, 1, 1152, 128))
    assert_size_stride(getitem_295, (401408, 32), (32, 1))
    assert_size_stride(getitem_296, (401408, ), (1, ))
    assert_size_stride(getitem_297, (401408, ), (1, ))
    assert_size_stride(rsqrt_23, (512, ), (1, ))
    assert_size_stride(getitem_302, (1605632, 32), (32, 1))
    assert_size_stride(getitem_303, (1605632, ), (1, ))
    assert_size_stride(getitem_304, (1605632, ), (1, ))
    assert_size_stride(getitem_307, (1605632, 16), (16, 1))
    assert_size_stride(convert_element_type_49, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convert_element_type_50, (2048, 512, 9, 9), (41472, 1, 4608, 512))
    assert_size_stride(getitem_308, (1605632, 32), (32, 1))
    assert_size_stride(getitem_309, (1605632, ), (1, ))
    assert_size_stride(getitem_310, (1605632, ), (1, ))
    assert_size_stride(rsqrt_24, (256, ), (1, ))
    assert_size_stride(getitem_315, (802816, 32), (32, 1))
    assert_size_stride(getitem_316, (802816, ), (1, ))
    assert_size_stride(getitem_317, (802816, ), (1, ))
    assert_size_stride(getitem_320, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_51, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convert_element_type_52, (2048, 256, 9, 9), (20736, 1, 2304, 256))
    assert_size_stride(getitem_321, (802816, 32), (32, 1))
    assert_size_stride(getitem_322, (802816, ), (1, ))
    assert_size_stride(getitem_323, (802816, ), (1, ))
    assert_size_stride(rsqrt_25, (256, ), (1, ))
    assert_size_stride(getitem_328, (200704, 32), (32, 1))
    assert_size_stride(getitem_329, (200704, ), (1, ))
    assert_size_stride(getitem_330, (200704, ), (1, ))
    assert_size_stride(getitem_333, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_53, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convert_element_type_54, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_334, (200704, 32), (32, 1))
    assert_size_stride(getitem_335, (200704, ), (1, ))
    assert_size_stride(getitem_336, (200704, ), (1, ))
    assert_size_stride(rsqrt_26, (1024, ), (1, ))
    assert_size_stride(getitem_341, (802816, 32), (32, 1))
    assert_size_stride(getitem_342, (802816, ), (1, ))
    assert_size_stride(getitem_343, (802816, ), (1, ))
    assert_size_stride(convert_element_type_55, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_344, (1605632, 32), (32, 1))
    assert_size_stride(getitem_345, (1605632, ), (1, ))
    assert_size_stride(getitem_346, (1605632, ), (1, ))
    assert_size_stride(rsqrt_27, (1024, ), (1, ))
    assert_size_stride(getitem_351, (802816, 32), (32, 1))
    assert_size_stride(getitem_352, (802816, ), (1, ))
    assert_size_stride(getitem_353, (802816, ), (1, ))
    assert_size_stride(getitem_356, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_57, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convert_element_type_58, (2048, 1024, 4, 4), (16384, 1, 4096, 1024))
    assert_size_stride(getitem_357, (802816, 32), (32, 1))
    assert_size_stride(getitem_358, (802816, ), (1, ))
    assert_size_stride(getitem_359, (802816, ), (1, ))
    assert_size_stride(rsqrt_28, (256, ), (1, ))
    assert_size_stride(getitem_364, (200704, 32), (32, 1))
    assert_size_stride(getitem_365, (200704, ), (1, ))
    assert_size_stride(getitem_366, (200704, ), (1, ))
    assert_size_stride(getitem_369, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_59, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convert_element_type_60, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_370, (200704, 32), (32, 1))
    assert_size_stride(getitem_371, (200704, ), (1, ))
    assert_size_stride(getitem_372, (200704, ), (1, ))
    assert_size_stride(rsqrt_29, (256, ), (1, ))
    assert_size_stride(getitem_377, (200704, 32), (32, 1))
    assert_size_stride(getitem_378, (200704, ), (1, ))
    assert_size_stride(getitem_379, (200704, ), (1, ))
    assert_size_stride(getitem_382, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_61, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convert_element_type_62, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_383, (200704, 32), (32, 1))
    assert_size_stride(getitem_384, (200704, ), (1, ))
    assert_size_stride(getitem_385, (200704, ), (1, ))
    assert_size_stride(rsqrt_30, (1024, ), (1, ))
    assert_size_stride(getitem_390, (802816, 32), (32, 1))
    assert_size_stride(getitem_391, (802816, ), (1, ))
    assert_size_stride(getitem_392, (802816, ), (1, ))
    assert_size_stride(getitem_395, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_63, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convert_element_type_64, (2048, 1024, 4, 4), (16384, 1, 4096, 1024))
    assert_size_stride(getitem_396, (802816, 32), (32, 1))
    assert_size_stride(getitem_397, (802816, ), (1, ))
    assert_size_stride(getitem_398, (802816, ), (1, ))
    assert_size_stride(rsqrt_31, (256, ), (1, ))
    assert_size_stride(getitem_403, (200704, 32), (32, 1))
    assert_size_stride(getitem_404, (200704, ), (1, ))
    assert_size_stride(getitem_405, (200704, ), (1, ))
    assert_size_stride(getitem_408, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_65, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convert_element_type_66, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_409, (200704, 32), (32, 1))
    assert_size_stride(getitem_410, (200704, ), (1, ))
    assert_size_stride(getitem_411, (200704, ), (1, ))
    assert_size_stride(rsqrt_32, (256, ), (1, ))
    assert_size_stride(getitem_416, (200704, 32), (32, 1))
    assert_size_stride(getitem_417, (200704, ), (1, ))
    assert_size_stride(getitem_418, (200704, ), (1, ))
    assert_size_stride(getitem_421, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_67, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convert_element_type_68, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_422, (200704, 32), (32, 1))
    assert_size_stride(getitem_423, (200704, ), (1, ))
    assert_size_stride(getitem_424, (200704, ), (1, ))
    assert_size_stride(rsqrt_33, (1024, ), (1, ))
    assert_size_stride(getitem_429, (802816, 32), (32, 1))
    assert_size_stride(getitem_430, (802816, ), (1, ))
    assert_size_stride(getitem_431, (802816, ), (1, ))
    assert_size_stride(getitem_434, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_69, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convert_element_type_70, (2048, 1024, 4, 4), (16384, 1, 4096, 1024))
    assert_size_stride(getitem_435, (802816, 32), (32, 1))
    assert_size_stride(getitem_436, (802816, ), (1, ))
    assert_size_stride(getitem_437, (802816, ), (1, ))
    assert_size_stride(rsqrt_34, (256, ), (1, ))
    assert_size_stride(getitem_442, (200704, 32), (32, 1))
    assert_size_stride(getitem_443, (200704, ), (1, ))
    assert_size_stride(getitem_444, (200704, ), (1, ))
    assert_size_stride(getitem_447, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_71, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convert_element_type_72, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_448, (200704, 32), (32, 1))
    assert_size_stride(getitem_449, (200704, ), (1, ))
    assert_size_stride(getitem_450, (200704, ), (1, ))
    assert_size_stride(rsqrt_35, (256, ), (1, ))
    assert_size_stride(getitem_455, (200704, 32), (32, 1))
    assert_size_stride(getitem_456, (200704, ), (1, ))
    assert_size_stride(getitem_457, (200704, ), (1, ))
    assert_size_stride(getitem_460, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_73, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convert_element_type_74, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_461, (200704, 32), (32, 1))
    assert_size_stride(getitem_462, (200704, ), (1, ))
    assert_size_stride(getitem_463, (200704, ), (1, ))
    assert_size_stride(rsqrt_36, (1024, ), (1, ))
    assert_size_stride(getitem_468, (802816, 32), (32, 1))
    assert_size_stride(getitem_469, (802816, ), (1, ))
    assert_size_stride(getitem_470, (802816, ), (1, ))
    assert_size_stride(getitem_473, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_75, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convert_element_type_76, (2048, 1024, 4, 4), (16384, 1, 4096, 1024))
    assert_size_stride(getitem_474, (802816, 32), (32, 1))
    assert_size_stride(getitem_475, (802816, ), (1, ))
    assert_size_stride(getitem_476, (802816, ), (1, ))
    assert_size_stride(rsqrt_37, (256, ), (1, ))
    assert_size_stride(getitem_481, (200704, 32), (32, 1))
    assert_size_stride(getitem_482, (200704, ), (1, ))
    assert_size_stride(getitem_483, (200704, ), (1, ))
    assert_size_stride(getitem_486, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_77, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convert_element_type_78, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_487, (200704, 32), (32, 1))
    assert_size_stride(getitem_488, (200704, ), (1, ))
    assert_size_stride(getitem_489, (200704, ), (1, ))
    assert_size_stride(rsqrt_38, (256, ), (1, ))
    assert_size_stride(getitem_494, (200704, 32), (32, 1))
    assert_size_stride(getitem_495, (200704, ), (1, ))
    assert_size_stride(getitem_496, (200704, ), (1, ))
    assert_size_stride(getitem_499, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_79, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convert_element_type_80, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_500, (200704, 32), (32, 1))
    assert_size_stride(getitem_501, (200704, ), (1, ))
    assert_size_stride(getitem_502, (200704, ), (1, ))
    assert_size_stride(rsqrt_39, (1024, ), (1, ))
    assert_size_stride(getitem_507, (802816, 32), (32, 1))
    assert_size_stride(getitem_508, (802816, ), (1, ))
    assert_size_stride(getitem_509, (802816, ), (1, ))
    assert_size_stride(getitem_512, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_81, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convert_element_type_82, (2048, 1024, 4, 4), (16384, 1, 4096, 1024))
    assert_size_stride(getitem_513, (802816, 32), (32, 1))
    assert_size_stride(getitem_514, (802816, ), (1, ))
    assert_size_stride(getitem_515, (802816, ), (1, ))
    assert_size_stride(rsqrt_40, (256, ), (1, ))
    assert_size_stride(getitem_520, (200704, 32), (32, 1))
    assert_size_stride(getitem_521, (200704, ), (1, ))
    assert_size_stride(getitem_522, (200704, ), (1, ))
    assert_size_stride(getitem_525, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_83, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convert_element_type_84, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_526, (200704, 32), (32, 1))
    assert_size_stride(getitem_527, (200704, ), (1, ))
    assert_size_stride(getitem_528, (200704, ), (1, ))
    assert_size_stride(rsqrt_41, (256, ), (1, ))
    assert_size_stride(getitem_533, (200704, 32), (32, 1))
    assert_size_stride(getitem_534, (200704, ), (1, ))
    assert_size_stride(getitem_535, (200704, ), (1, ))
    assert_size_stride(getitem_538, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_85, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convert_element_type_86, (2048, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(getitem_539, (200704, 32), (32, 1))
    assert_size_stride(getitem_540, (200704, ), (1, ))
    assert_size_stride(getitem_541, (200704, ), (1, ))
    assert_size_stride(rsqrt_42, (1024, ), (1, ))
    assert_size_stride(getitem_546, (802816, 32), (32, 1))
    assert_size_stride(getitem_547, (802816, ), (1, ))
    assert_size_stride(getitem_548, (802816, ), (1, ))
    assert_size_stride(getitem_551, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_87, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convert_element_type_88, (2048, 1024, 4, 4), (16384, 1, 4096, 1024))
    assert_size_stride(getitem_552, (802816, 32), (32, 1))
    assert_size_stride(getitem_553, (802816, ), (1, ))
    assert_size_stride(getitem_554, (802816, ), (1, ))
    assert_size_stride(rsqrt_43, (512, ), (1, ))
    assert_size_stride(getitem_559, (401408, 32), (32, 1))
    assert_size_stride(getitem_560, (401408, ), (1, ))
    assert_size_stride(getitem_561, (401408, ), (1, ))
    assert_size_stride(getitem_564, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_89, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(convert_element_type_90, (2048, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(getitem_565, (401408, 32), (32, 1))
    assert_size_stride(getitem_566, (401408, ), (1, ))
    assert_size_stride(getitem_567, (401408, ), (1, ))
    assert_size_stride(rsqrt_44, (512, ), (1, ))
    assert_size_stride(getitem_572, (100352, 32), (32, 1))
    assert_size_stride(getitem_573, (100352, ), (1, ))
    assert_size_stride(getitem_574, (100352, ), (1, ))
    assert_size_stride(getitem_577, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_91, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convert_element_type_92, (2048, 512, 2, 2), (2048, 1, 1024, 512))
    assert_size_stride(getitem_578, (100352, 32), (32, 1))
    assert_size_stride(getitem_579, (100352, ), (1, ))
    assert_size_stride(getitem_580, (100352, ), (1, ))
    assert_size_stride(rsqrt_45, (2048, ), (1, ))
    assert_size_stride(getitem_585, (401408, 32), (32, 1))
    assert_size_stride(getitem_586, (401408, ), (1, ))
    assert_size_stride(getitem_587, (401408, ), (1, ))
    assert_size_stride(convert_element_type_93, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_588, (802816, 32), (32, 1))
    assert_size_stride(getitem_589, (802816, ), (1, ))
    assert_size_stride(getitem_590, (802816, ), (1, ))
    assert_size_stride(rsqrt_46, (2048, ), (1, ))
    assert_size_stride(getitem_595, (401408, 32), (32, 1))
    assert_size_stride(getitem_596, (401408, ), (1, ))
    assert_size_stride(getitem_597, (401408, ), (1, ))
    assert_size_stride(getitem_600, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_95, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(convert_element_type_96, (2048, 2048, 2, 2), (8192, 1, 4096, 2048))
    assert_size_stride(getitem_601, (401408, 32), (32, 1))
    assert_size_stride(getitem_602, (401408, ), (1, ))
    assert_size_stride(getitem_603, (401408, ), (1, ))
    assert_size_stride(rsqrt_47, (512, ), (1, ))
    assert_size_stride(getitem_608, (100352, 32), (32, 1))
    assert_size_stride(getitem_609, (100352, ), (1, ))
    assert_size_stride(getitem_610, (100352, ), (1, ))
    assert_size_stride(getitem_613, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_97, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(convert_element_type_98, (2048, 512, 2, 2), (2048, 1, 1024, 512))
    assert_size_stride(getitem_614, (100352, 32), (32, 1))
    assert_size_stride(getitem_615, (100352, ), (1, ))
    assert_size_stride(getitem_616, (100352, ), (1, ))
    assert_size_stride(rsqrt_48, (512, ), (1, ))
    assert_size_stride(getitem_621, (100352, 32), (32, 1))
    assert_size_stride(getitem_622, (100352, ), (1, ))
    assert_size_stride(getitem_623, (100352, ), (1, ))
    assert_size_stride(getitem_626, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_99, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convert_element_type_100, (2048, 512, 2, 2), (2048, 1, 1024, 512))
    assert_size_stride(getitem_627, (100352, 32), (32, 1))
    assert_size_stride(getitem_628, (100352, ), (1, ))
    assert_size_stride(getitem_629, (100352, ), (1, ))
    assert_size_stride(rsqrt_49, (2048, ), (1, ))
    assert_size_stride(getitem_634, (401408, 32), (32, 1))
    assert_size_stride(getitem_635, (401408, ), (1, ))
    assert_size_stride(getitem_636, (401408, ), (1, ))
    assert_size_stride(getitem_639, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_101, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(convert_element_type_102, (2048, 2048, 2, 2), (8192, 1, 4096, 2048))
    assert_size_stride(getitem_640, (401408, 32), (32, 1))
    assert_size_stride(getitem_641, (401408, ), (1, ))
    assert_size_stride(getitem_642, (401408, ), (1, ))
    assert_size_stride(rsqrt_50, (512, ), (1, ))
    assert_size_stride(getitem_647, (100352, 32), (32, 1))
    assert_size_stride(getitem_648, (100352, ), (1, ))
    assert_size_stride(getitem_649, (100352, ), (1, ))
    assert_size_stride(getitem_652, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_103, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(convert_element_type_104, (2048, 512, 2, 2), (2048, 1, 1024, 512))
    assert_size_stride(getitem_653, (100352, 32), (32, 1))
    assert_size_stride(getitem_654, (100352, ), (1, ))
    assert_size_stride(getitem_655, (100352, ), (1, ))
    assert_size_stride(rsqrt_51, (512, ), (1, ))
    assert_size_stride(getitem_660, (100352, 32), (32, 1))
    assert_size_stride(getitem_661, (100352, ), (1, ))
    assert_size_stride(getitem_662, (100352, ), (1, ))
    assert_size_stride(getitem_665, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_105, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convert_element_type_106, (2048, 512, 2, 2), (2048, 1, 1024, 512))
    assert_size_stride(getitem_666, (100352, 32), (32, 1))
    assert_size_stride(getitem_667, (100352, ), (1, ))
    assert_size_stride(getitem_668, (100352, ), (1, ))
    assert_size_stride(rsqrt_52, (2048, ), (1, ))
    assert_size_stride(getitem_673, (401408, 32), (32, 1))
    assert_size_stride(getitem_674, (401408, ), (1, ))
    assert_size_stride(getitem_675, (401408, ), (1, ))
    assert_size_stride(getitem_678, (401408, 16), (16, 1))
    assert_size_stride(getitem_679, (8192, 32), (32, 1))
    assert_size_stride(getitem_680, (8192, ), (1, ))
    assert_size_stride(getitem_681, (8192, ), (1, ))
    assert_size_stride(convert_element_type_107, (100, 2048), (2048, 1))
    assert_size_stride(tangents_1, (2048, 100), (100, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_679, getitem_680, getitem_681, buf0, 32, 1, 512, 1, 2, 16, 32, 8192, 1, 1, stream=stream0)
        del getitem_679
        del getitem_680
        del getitem_681
        buf2 = empty_strided_cuda((100, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (100, 2048), (1, 100), 0), reinterpret_tensor(buf0, (2048, 2048), (2048, 1), 0), out=buf2)
        buf3 = reinterpret_tensor(buf0, (2048, 2048), (2048, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [view_1329], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, convert_element_type_107, out=buf3)
        del convert_element_type_107
        del tangents_1
        buf4 = empty_strided_cuda((100, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(buf2, buf4, 204800, stream=stream0)
        del buf2
        buf5 = empty_strided_cuda((401408, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_257], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_1.run(buf5, 205520896, stream=stream0)
        buf6 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf5, buf6, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_678, reinterpret_tensor(buf6, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_678
        buf8 = empty_strided_cuda((401408, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_673, getitem_674, getitem_675, buf8, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_673
        del getitem_674
        del getitem_675
        buf10 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_3.run(buf10, 2048, stream=stream0)
        buf11 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf12 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(buf10, buf11, buf12, 2048, stream=stream0)
        buf13 = empty_strided_cuda((2048, 2048, 49), (100352, 49, 1), torch.bfloat16)
        buf17 = empty_strided_cuda((2048, 2048, 49), (100352, 49, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(buf3, buf6, buf13, buf17, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf8, (2048, 2048, 49), (100352, 49, 1), 0), buf13, buf11, buf12, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        buf16 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf8, (2048, 2048, 49), (100352, 49, 1), 0), buf17, rsqrt_52, primals_316, buf11, buf12, buf16, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        del primals_316
        del rsqrt_52
        buf19 = empty_strided_cuda((100352, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_666, getitem_667, getitem_668, buf19, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_666
        del getitem_667
        del getitem_668
        buf21 = empty_strided_cuda((1, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf22 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf16, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf21, (2048, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_105, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_105
        buf23 = buf22[0]
        assert_size_stride(buf23, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf23, 16, 'torch.ops.aten.convolution_backward.default')
        del buf22
        buf24 = buf21; del buf21  # reuse
        buf25 = empty_strided_cuda((2048, 512, 7, 7), (25088, 49, 7, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_6.run(buf19, convert_element_type_106, buf25, 51380224, stream=stream0)
        del convert_element_type_106
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf26 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf16, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), buf25, reinterpret_tensor(buf24, (2048, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf27 = buf26[1]
        assert_size_stride(buf27, (2048, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf27, 16, 'torch.ops.aten.convolution_backward.default')
        del buf26
        buf28 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf27, buf28, 1048576, stream=stream0)
        del buf27
        buf29 = empty_strided_cuda((100352, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_260], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_8.run(buf29, 51380224, stream=stream0)
        buf30 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf53 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(buf29, buf30, buf53, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_665, reinterpret_tensor(buf30, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_665
        buf32 = reinterpret_tensor(buf25, (100352, 512), (512, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_660, getitem_661, getitem_662, buf32, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_660
        del getitem_661
        del getitem_662
        buf34 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_10.run(buf34, 512, stream=stream0)
        buf35 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf36 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf57 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf58 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf34, buf35, buf36, buf57, buf58, 512, stream=stream0)
        buf37 = reinterpret_tensor(buf19, (2048, 512, 49), (25088, 49, 1), 0); del buf19  # reuse
        buf41 = empty_strided_cuda((2048, 512, 49), (25088, 49, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf23, buf30, buf37, buf41, 1048576, 49, stream=stream0)
        del buf23
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf32, (2048, 512, 49), (25088, 49, 1), 0), buf37, buf35, buf36, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf40 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf32, (2048, 512, 49), (25088, 49, 1), 0), buf41, rsqrt_51, primals_310, buf35, buf36, buf40, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del primals_310
        del rsqrt_51
        buf43 = reinterpret_tensor(buf41, (100352, 512), (512, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_653, getitem_654, getitem_655, buf43, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_653
        del getitem_654
        del getitem_655
        buf45 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf46 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf40, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf45, (2048, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_103, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_103
        buf47 = buf46[0]
        assert_size_stride(buf47, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf47, 16, 'torch.ops.aten.convolution_backward.default')
        del buf46
        buf48 = buf45; del buf45  # reuse
        buf49 = reinterpret_tensor(buf32, (2048, 512, 7, 7), (25088, 49, 7, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_6.run(buf43, convert_element_type_104, buf49, 51380224, stream=stream0)
        del convert_element_type_104
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf50 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf40, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf49, reinterpret_tensor(buf48, (512, 512, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf51 = buf50[1]
        assert_size_stride(buf51, (512, 512, 3, 3), (4608, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf51, 16, 'torch.ops.aten.convolution_backward.default')
        del buf50
        buf52 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf51, buf52, 2359296, stream=stream0)
        del buf51
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_652, reinterpret_tensor(buf53, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_652
        buf55 = reinterpret_tensor(buf49, (100352, 512), (512, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_647, getitem_648, getitem_649, buf55, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_647
        del getitem_648
        del getitem_649
        buf59 = buf40; del buf40  # reuse
        buf63 = reinterpret_tensor(buf43, (2048, 512, 49), (25088, 49, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf47, buf53, buf59, buf63, 1048576, 49, stream=stream0)
        del buf47
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf55, (2048, 512, 49), (25088, 49, 1), 0), buf59, buf57, buf58, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf62 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf55, (2048, 512, 49), (25088, 49, 1), 0), buf63, rsqrt_50, primals_304, buf57, buf58, buf62, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del primals_304
        del rsqrt_50
        buf65 = reinterpret_tensor(buf16, (401408, 512), (512, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_640, getitem_641, getitem_642, buf65, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_640
        del getitem_641
        del getitem_642
        buf67 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf68 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf62, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf67, (2048, 2048, 7, 7), (0, 0, 0, 0), 0), convert_element_type_101, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_101
        buf69 = buf68[0]
        assert_size_stride(buf69, (2048, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf69, 16, 'torch.ops.aten.convolution_backward.default')
        del buf68
        buf70 = buf67; del buf67  # reuse
        buf71 = reinterpret_tensor(buf8, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_14.run(buf65, convert_element_type_102, buf71, 205520896, stream=stream0)
        del convert_element_type_102
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf72 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf62, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf71, reinterpret_tensor(buf70, (512, 2048, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf73 = buf72[1]
        assert_size_stride(buf73, (512, 2048, 1, 1), (2048, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf73, 16, 'torch.ops.aten.convolution_backward.default')
        del buf72
        buf74 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf73, buf74, 1048576, stream=stream0)
        del buf73
        buf75 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf5, buf75, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_639, reinterpret_tensor(buf75, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_639
        buf77 = reinterpret_tensor(buf71, (401408, 512), (512, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_634, getitem_635, getitem_636, buf77, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_634
        del getitem_635
        del getitem_636
        buf79 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf80 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(buf10, buf79, buf80, 2048, stream=stream0)
        buf81 = reinterpret_tensor(buf65, (2048, 2048, 49), (100352, 49, 1), 0); del buf65  # reuse
        buf85 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(buf3, buf6, buf69, buf75, buf81, buf85, 4194304, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf77, (2048, 2048, 49), (100352, 49, 1), 0), buf81, buf79, buf80, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        buf84 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf77, (2048, 2048, 49), (100352, 49, 1), 0), buf85, rsqrt_49, primals_298, buf79, buf80, buf84, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        del buf77
        del primals_298
        del rsqrt_49
        buf87 = reinterpret_tensor(buf62, (100352, 512), (512, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_627, getitem_628, getitem_629, buf87, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_627
        del getitem_628
        del getitem_629
        buf89 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf90 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf84, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf89, (2048, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_99, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_99
        buf91 = buf90[0]
        assert_size_stride(buf91, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf91, 16, 'torch.ops.aten.convolution_backward.default')
        del buf90
        buf92 = buf89; del buf89  # reuse
        buf93 = reinterpret_tensor(buf63, (2048, 512, 7, 7), (25088, 49, 7, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_6.run(buf87, convert_element_type_100, buf93, 51380224, stream=stream0)
        del convert_element_type_100
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf94 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf84, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), buf93, reinterpret_tensor(buf92, (2048, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf95 = buf94[1]
        assert_size_stride(buf95, (2048, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf95, 16, 'torch.ops.aten.convolution_backward.default')
        del buf94
        buf96 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf95, buf96, 1048576, stream=stream0)
        del buf95
        buf97 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf119 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(buf29, buf97, buf119, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_626, reinterpret_tensor(buf97, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_626
        buf99 = reinterpret_tensor(buf93, (100352, 512), (512, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_621, getitem_622, getitem_623, buf99, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_621
        del getitem_622
        del getitem_623
        buf101 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf102 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf123 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf124 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf34, buf101, buf102, buf123, buf124, 512, stream=stream0)
        buf103 = reinterpret_tensor(buf87, (2048, 512, 49), (25088, 49, 1), 0); del buf87  # reuse
        buf107 = reinterpret_tensor(buf55, (2048, 512, 49), (25088, 49, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf91, buf97, buf103, buf107, 1048576, 49, stream=stream0)
        del buf91
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf99, (2048, 512, 49), (25088, 49, 1), 0), buf103, buf101, buf102, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf106 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf99, (2048, 512, 49), (25088, 49, 1), 0), buf107, rsqrt_48, primals_292, buf101, buf102, buf106, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del primals_292
        del rsqrt_48
        buf109 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_614, getitem_615, getitem_616, buf109, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_614
        del getitem_615
        del getitem_616
        buf111 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf112 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf106, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf111, (2048, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_97, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_97
        buf113 = buf112[0]
        assert_size_stride(buf113, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf113, 16, 'torch.ops.aten.convolution_backward.default')
        del buf112
        buf114 = buf111; del buf111  # reuse
        buf115 = reinterpret_tensor(buf107, (2048, 512, 7, 7), (25088, 49, 7, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_6.run(buf109, convert_element_type_98, buf115, 51380224, stream=stream0)
        del convert_element_type_98
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf116 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf106, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf115, reinterpret_tensor(buf114, (512, 512, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf117 = buf116[1]
        assert_size_stride(buf117, (512, 512, 3, 3), (4608, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf117, 16, 'torch.ops.aten.convolution_backward.default')
        del buf116
        buf118 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf117, buf118, 2359296, stream=stream0)
        del buf117
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_613, reinterpret_tensor(buf119, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_613
        buf121 = reinterpret_tensor(buf115, (100352, 512), (512, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_608, getitem_609, getitem_610, buf121, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_608
        del getitem_609
        del getitem_610
        buf125 = buf106; del buf106  # reuse
        buf129 = reinterpret_tensor(buf109, (2048, 512, 49), (25088, 49, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf113, buf119, buf125, buf129, 1048576, 49, stream=stream0)
        del buf113
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf121, (2048, 512, 49), (25088, 49, 1), 0), buf125, buf123, buf124, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf128 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf121, (2048, 512, 49), (25088, 49, 1), 0), buf129, rsqrt_47, primals_286, buf123, buf124, buf128, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del primals_286
        del rsqrt_47
        buf131 = reinterpret_tensor(buf84, (401408, 512), (512, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_601, getitem_602, getitem_603, buf131, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_601
        del getitem_602
        del getitem_603
        buf133 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf134 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf128, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf133, (2048, 2048, 7, 7), (0, 0, 0, 0), 0), convert_element_type_95, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_95
        buf135 = buf134[0]
        assert_size_stride(buf135, (2048, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf135, 16, 'torch.ops.aten.convolution_backward.default')
        del buf134
        buf136 = buf133; del buf133  # reuse
        buf137 = reinterpret_tensor(buf85, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_14.run(buf131, convert_element_type_96, buf137, 205520896, stream=stream0)
        del convert_element_type_96
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf138 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf128, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf137, reinterpret_tensor(buf136, (512, 2048, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf139 = buf138[1]
        assert_size_stride(buf139, (512, 2048, 1, 1), (2048, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf139, 16, 'torch.ops.aten.convolution_backward.default')
        del buf138
        buf140 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf139, buf140, 1048576, stream=stream0)
        del buf139
        buf141 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        buf200 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(buf5, buf141, buf200, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_600, reinterpret_tensor(buf141, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_600
        buf143 = reinterpret_tensor(buf137, (401408, 512), (512, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_595, getitem_596, getitem_597, buf143, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_595
        del getitem_596
        del getitem_597
        buf145 = reinterpret_tensor(buf131, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_17.run(buf3, buf6, buf69, buf75, buf135, buf141, buf145, 4194304, 49, stream=stream0)
        del buf135
        del buf3
        buf146 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf147 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf164 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(buf10, buf146, buf147, buf164, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf143, (2048, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf145, (2048, 2048, 49), (100352, 49, 1), 0), buf146, buf147, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        buf150 = reinterpret_tensor(buf69, (2048, 2048, 49), (100352, 49, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf143, (2048, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf145, (2048, 2048, 49), (100352, 49, 1), 0), rsqrt_46, primals_280, buf146, buf147, buf150, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        del primals_280
        del rsqrt_46
        buf152 = empty_strided_cuda((802816, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_588, getitem_589, getitem_590, buf152, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_588
        del getitem_589
        del getitem_590
        buf154 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf155 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf150, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf154, (2048, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_93, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_93
        buf156 = buf155[0]
        assert_size_stride(buf156, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf156, 16, 'torch.ops.aten.convolution_backward.default')
        del buf155
        buf157 = buf154; del buf154  # reuse
        buf212 = empty_strided_cuda((802816, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_552, getitem_553, getitem_554, buf212, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_552
        del getitem_553
        del getitem_554
        buf158 = empty_strided_cuda((2048, 1024, 14, 14), (200704, 196, 14, 1), torch.bfloat16)
        buf218 = empty_strided_cuda((2048, 1024, 14, 14), (200704, 196, 14, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_19.run(buf152, convert_element_type_88, buf212, buf158, buf218, 411041792, stream=stream0)
        del buf152
        del convert_element_type_88
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf159 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf150, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), buf158, reinterpret_tensor(buf157, (2048, 1024, 1, 1), (0, 0, 0, 0), 0), None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf160 = buf159[1]
        assert_size_stride(buf160, (2048, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf160, 16, 'torch.ops.aten.convolution_backward.default')
        del buf159
        buf161 = empty_strided_cuda((2048, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_20.run(buf160, buf161, 2097152, stream=stream0)
        del buf160
        buf162 = reinterpret_tensor(buf150, (401408, 512), (512, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_585, getitem_586, getitem_587, buf162, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_585
        del getitem_586
        del getitem_587
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf162, (2048, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf145, (2048, 2048, 49), (100352, 49, 1), 0), buf10, buf164, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        buf167 = reinterpret_tensor(buf143, (2048, 2048, 49), (100352, 49, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf162, (2048, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf145, (2048, 2048, 49), (100352, 49, 1), 0), rsqrt_45, primals_274, buf10, buf164, buf167, 100352, 49, 100352, 49, 1024, 2048, 98, 1, stream=stream0)
        del primals_274
        del rsqrt_45
        buf169 = reinterpret_tensor(buf128, (100352, 512), (512, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_578, getitem_579, getitem_580, buf169, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_578
        del getitem_579
        del getitem_580
        buf171 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf172 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf167, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf171, (2048, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_91, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_91
        buf173 = buf172[0]
        assert_size_stride(buf173, (2048, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf173, 16, 'torch.ops.aten.convolution_backward.default')
        del buf172
        buf174 = buf171; del buf171  # reuse
        buf175 = reinterpret_tensor(buf129, (2048, 512, 7, 7), (25088, 49, 7, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_6.run(buf169, convert_element_type_92, buf175, 51380224, stream=stream0)
        del convert_element_type_92
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf176 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf167, (2048, 2048, 7, 7), (100352, 49, 7, 1), 0), buf175, reinterpret_tensor(buf174, (2048, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf177 = buf176[1]
        assert_size_stride(buf177, (2048, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf177, 16, 'torch.ops.aten.convolution_backward.default')
        del buf176
        buf178 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf177, buf178, 1048576, stream=stream0)
        del buf177
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_577, buf29, 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del buf119
        del buf30
        del buf53
        del buf97
        del getitem_577
        buf180 = reinterpret_tensor(buf175, (100352, 512), (512, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_572, getitem_573, getitem_574, buf180, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_572
        del getitem_573
        del getitem_574
        buf182 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf183 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf204 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf205 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf34, buf182, buf183, buf204, buf205, 512, stream=stream0)
        buf184 = reinterpret_tensor(buf169, (2048, 512, 49), (25088, 49, 1), 0); del buf169  # reuse
        buf188 = reinterpret_tensor(buf121, (2048, 512, 49), (25088, 49, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf173, buf29, buf184, buf188, 1048576, 49, stream=stream0)
        del buf173
        del buf29
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf180, (2048, 512, 49), (25088, 49, 1), 0), buf184, buf182, buf183, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        buf187 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf180, (2048, 512, 49), (25088, 49, 1), 0), buf188, rsqrt_44, primals_268, buf182, buf183, buf187, 100352, 49, 25088, 49, 1024, 512, 98, 1, stream=stream0)
        del buf180
        del buf188
        del primals_268
        del rsqrt_44
        buf190 = reinterpret_tensor(buf167, (401408, 512), (512, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_565, getitem_566, getitem_567, buf190, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_565
        del getitem_566
        del getitem_567
        buf192 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf193 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf187, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf192, (2048, 512, 14, 14), (0, 0, 0, 0), 0), convert_element_type_89, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_89
        buf194 = buf193[0]
        assert_size_stride(buf194, (2048, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf194, 16, 'torch.ops.aten.convolution_backward.default')
        del buf193
        buf195 = buf192; del buf192  # reuse
        buf196 = reinterpret_tensor(buf162, (2048, 512, 14, 14), (100352, 196, 14, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_21.run(buf190, convert_element_type_90, buf196, 205520896, stream=stream0)
        del convert_element_type_90
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf197 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf187, (2048, 512, 7, 7), (25088, 49, 7, 1), 0), buf196, reinterpret_tensor(buf195, (512, 512, 3, 3), (0, 0, 0, 0), 0), None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        del buf187
        buf198 = buf197[1]
        assert_size_stride(buf198, (512, 512, 3, 3), (4608, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf198, 16, 'torch.ops.aten.convolution_backward.default')
        del buf197
        buf199 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf198, buf199, 2359296, stream=stream0)
        del buf198
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_564, reinterpret_tensor(buf200, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_564
        buf202 = reinterpret_tensor(buf196, (401408, 512), (512, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_559, getitem_560, getitem_561, buf202, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_559
        del getitem_560
        del getitem_561
        buf206 = reinterpret_tensor(buf190, (2048, 512, 196), (100352, 196, 1), 0); del buf190  # reuse
        buf210 = reinterpret_tensor(buf145, (2048, 512, 196), (100352, 196, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_22.run(buf194, buf200, buf206, buf210, 1048576, 196, stream=stream0)
        del buf194
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_6.run(reinterpret_tensor(buf202, (2048, 512, 196), (100352, 196, 1), 0), buf206, buf204, buf205, 401408, 196, 100352, 196, 1024, 512, 392, 1, stream=stream0)
        buf209 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_7.run(reinterpret_tensor(buf202, (2048, 512, 196), (100352, 196, 1), 0), buf210, rsqrt_43, primals_262, buf204, buf205, buf209, 401408, 196, 100352, 196, 1024, 512, 392, 1, stream=stream0)
        del primals_262
        del rsqrt_43
        buf214 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf215 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf209, (2048, 512, 14, 14), (100352, 196, 14, 1), 0), reinterpret_tensor(buf214, (2048, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_87, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_87
        buf216 = buf215[0]
        assert_size_stride(buf216, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf216, 16, 'torch.ops.aten.convolution_backward.default')
        del buf215
        buf217 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf219 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf209, (2048, 512, 14, 14), (100352, 196, 14, 1), 0), buf218, reinterpret_tensor(buf217, (512, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf220 = buf219[1]
        assert_size_stride(buf220, (512, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf220, 16, 'torch.ops.aten.convolution_backward.default')
        del buf219
        buf221 = empty_strided_cuda((512, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_23.run(buf220, buf221, 524288, stream=stream0)
        del buf220
        buf222 = empty_strided_cuda((802816, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_286], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_24.run(buf222, 411041792, stream=stream0)
        buf223 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_25.run(buf222, buf223, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_551, reinterpret_tensor(buf223, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_551
        buf225 = reinterpret_tensor(buf218, (802816, 512), (512, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_546, getitem_547, getitem_548, buf225, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_546
        del getitem_547
        del getitem_548
        buf227 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_26.run(buf227, 1024, stream=stream0)
        buf228 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf229 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_27.run(buf227, buf228, buf229, 1024, stream=stream0)
        buf230 = reinterpret_tensor(buf158, (2048, 1024, 196), (200704, 196, 1), 0); del buf158  # reuse
        buf234 = reinterpret_tensor(buf212, (2048, 1024, 196), (200704, 196, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf156, buf216, buf223, buf230, buf234, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf225, (2048, 1024, 196), (200704, 196, 1), 0), buf230, buf228, buf229, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf233 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf225, (2048, 1024, 196), (200704, 196, 1), 0), buf234, rsqrt_42, primals_256, buf228, buf229, buf233, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del buf225
        del primals_256
        del rsqrt_42
        buf236 = empty_strided_cuda((200704, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_539, getitem_540, getitem_541, buf236, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_539
        del getitem_540
        del getitem_541
        buf238 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf239 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf233, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf238, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_85, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_85
        buf240 = buf239[0]
        assert_size_stride(buf240, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf240, 16, 'torch.ops.aten.convolution_backward.default')
        del buf239
        buf241 = buf238; del buf238  # reuse
        buf242 = empty_strided_cuda((2048, 256, 14, 14), (50176, 196, 14, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf236, convert_element_type_86, buf242, 102760448, stream=stream0)
        del convert_element_type_86
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf243 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf233, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), buf242, reinterpret_tensor(buf241, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf244 = buf243[1]
        assert_size_stride(buf244, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf244, 16, 'torch.ops.aten.convolution_backward.default')
        del buf243
        buf245 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf244, buf245, 262144, stream=stream0)
        del buf244
        buf246 = empty_strided_cuda((200704, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_289], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_31.run(buf246, 102760448, stream=stream0)
        buf247 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf270 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf246, buf247, buf270, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_538, reinterpret_tensor(buf247, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_538
        buf249 = reinterpret_tensor(buf242, (200704, 512), (512, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_533, getitem_534, getitem_535, buf249, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_533
        del getitem_534
        del getitem_535
        buf251 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_33.run(buf251, 256, stream=stream0)
        buf252 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf253 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf274 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf275 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf251, buf252, buf253, buf274, buf275, 256, stream=stream0)
        buf254 = reinterpret_tensor(buf236, (2048, 256, 196), (50176, 196, 1), 0); del buf236  # reuse
        buf258 = empty_strided_cuda((2048, 256, 196), (50176, 196, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf240, buf247, buf254, buf258, 524288, 196, stream=stream0)
        del buf240
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf249, (2048, 256, 196), (50176, 196, 1), 0), buf254, buf252, buf253, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf257 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf249, (2048, 256, 196), (50176, 196, 1), 0), buf258, rsqrt_41, primals_250, buf252, buf253, buf257, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_250
        del rsqrt_41
        buf260 = reinterpret_tensor(buf258, (200704, 512), (512, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_526, getitem_527, getitem_528, buf260, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_526
        del getitem_527
        del getitem_528
        buf262 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf263 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf257, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf262, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_83, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_83
        buf264 = buf263[0]
        assert_size_stride(buf264, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf264, 16, 'torch.ops.aten.convolution_backward.default')
        del buf263
        buf265 = buf262; del buf262  # reuse
        buf266 = reinterpret_tensor(buf249, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf260, convert_element_type_84, buf266, 102760448, stream=stream0)
        del convert_element_type_84
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf267 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf257, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf266, reinterpret_tensor(buf265, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf268 = buf267[1]
        assert_size_stride(buf268, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf268, 16, 'torch.ops.aten.convolution_backward.default')
        del buf267
        buf269 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_36.run(buf268, buf269, 589824, stream=stream0)
        del buf268
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_525, reinterpret_tensor(buf270, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_525
        buf272 = reinterpret_tensor(buf266, (200704, 512), (512, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_520, getitem_521, getitem_522, buf272, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_520
        del getitem_521
        del getitem_522
        buf276 = buf257; del buf257  # reuse
        buf280 = reinterpret_tensor(buf260, (2048, 256, 196), (50176, 196, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf264, buf270, buf276, buf280, 524288, 196, stream=stream0)
        del buf264
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf272, (2048, 256, 196), (50176, 196, 1), 0), buf276, buf274, buf275, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf279 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf272, (2048, 256, 196), (50176, 196, 1), 0), buf280, rsqrt_40, primals_244, buf274, buf275, buf279, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_244
        del rsqrt_40
        buf282 = reinterpret_tensor(buf233, (802816, 512), (512, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_513, getitem_514, getitem_515, buf282, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_513
        del getitem_514
        del getitem_515
        buf284 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf285 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf279, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf284, (2048, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_81, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_81
        buf286 = buf285[0]
        assert_size_stride(buf286, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf286, 16, 'torch.ops.aten.convolution_backward.default')
        del buf285
        buf287 = buf284; del buf284  # reuse
        buf288 = reinterpret_tensor(buf234, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_37.run(buf282, convert_element_type_82, buf288, 411041792, stream=stream0)
        del buf282
        del convert_element_type_82
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf289 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf279, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf288, reinterpret_tensor(buf287, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf290 = buf289[1]
        assert_size_stride(buf290, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf290, 16, 'torch.ops.aten.convolution_backward.default')
        del buf289
        buf291 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf290, buf291, 262144, stream=stream0)
        del buf290
        buf292 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        buf357 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf222, buf292, buf357, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_512, reinterpret_tensor(buf292, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_512
        buf294 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_39.run(buf156, buf216, buf223, buf286, buf292, buf294, 2097152, 196, stream=stream0)
        buf295 = reinterpret_tensor(buf286, (802816, 512), (512, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_507, getitem_508, getitem_509, buf295, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_507
        del getitem_508
        del getitem_509
        buf297 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf298 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf361 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf362 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_40.run(buf227, buf297, buf298, buf361, buf362, 1024, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf295, (2048, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf294, (2048, 1024, 196), (200704, 196, 1), 0), buf297, buf298, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf301 = reinterpret_tensor(buf216, (2048, 1024, 196), (200704, 196, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf295, (2048, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf294, (2048, 1024, 196), (200704, 196, 1), 0), rsqrt_39, primals_238, buf297, buf298, buf301, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_238
        del rsqrt_39
        buf303 = reinterpret_tensor(buf279, (200704, 512), (512, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_500, getitem_501, getitem_502, buf303, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_500
        del getitem_501
        del getitem_502
        buf305 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf306 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf301, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf305, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_79, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_79
        buf307 = buf306[0]
        assert_size_stride(buf307, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf307, 16, 'torch.ops.aten.convolution_backward.default')
        del buf306
        buf308 = buf305; del buf305  # reuse
        buf309 = reinterpret_tensor(buf280, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf303, convert_element_type_80, buf309, 102760448, stream=stream0)
        del convert_element_type_80
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf310 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf301, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), buf309, reinterpret_tensor(buf308, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf311 = buf310[1]
        assert_size_stride(buf311, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf311, 16, 'torch.ops.aten.convolution_backward.default')
        del buf310
        buf312 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf311, buf312, 262144, stream=stream0)
        del buf311
        buf313 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf335 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf246, buf313, buf335, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_499, reinterpret_tensor(buf313, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_499
        buf315 = reinterpret_tensor(buf309, (200704, 512), (512, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_494, getitem_495, getitem_496, buf315, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_494
        del getitem_495
        del getitem_496
        buf317 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf318 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf339 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf340 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf251, buf317, buf318, buf339, buf340, 256, stream=stream0)
        buf319 = reinterpret_tensor(buf303, (2048, 256, 196), (50176, 196, 1), 0); del buf303  # reuse
        buf323 = reinterpret_tensor(buf272, (2048, 256, 196), (50176, 196, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf307, buf313, buf319, buf323, 524288, 196, stream=stream0)
        del buf307
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf315, (2048, 256, 196), (50176, 196, 1), 0), buf319, buf317, buf318, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf322 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf315, (2048, 256, 196), (50176, 196, 1), 0), buf323, rsqrt_38, primals_232, buf317, buf318, buf322, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_232
        del rsqrt_38
        buf325 = reinterpret_tensor(buf323, (200704, 512), (512, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_487, getitem_488, getitem_489, buf325, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_487
        del getitem_488
        del getitem_489
        buf327 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf328 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf322, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf327, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_77, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_77
        buf329 = buf328[0]
        assert_size_stride(buf329, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf329, 16, 'torch.ops.aten.convolution_backward.default')
        del buf328
        buf330 = buf327; del buf327  # reuse
        buf331 = reinterpret_tensor(buf315, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf325, convert_element_type_78, buf331, 102760448, stream=stream0)
        del convert_element_type_78
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf332 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf322, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf331, reinterpret_tensor(buf330, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf333 = buf332[1]
        assert_size_stride(buf333, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf333, 16, 'torch.ops.aten.convolution_backward.default')
        del buf332
        buf334 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_36.run(buf333, buf334, 589824, stream=stream0)
        del buf333
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_486, reinterpret_tensor(buf335, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_486
        buf337 = reinterpret_tensor(buf331, (200704, 512), (512, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_481, getitem_482, getitem_483, buf337, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_481
        del getitem_482
        del getitem_483
        buf341 = buf322; del buf322  # reuse
        buf345 = reinterpret_tensor(buf325, (2048, 256, 196), (50176, 196, 1), 0); del buf325  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf329, buf335, buf341, buf345, 524288, 196, stream=stream0)
        del buf329
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf337, (2048, 256, 196), (50176, 196, 1), 0), buf341, buf339, buf340, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf344 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf337, (2048, 256, 196), (50176, 196, 1), 0), buf345, rsqrt_37, primals_226, buf339, buf340, buf344, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_226
        del rsqrt_37
        buf347 = reinterpret_tensor(buf301, (802816, 512), (512, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_474, getitem_475, getitem_476, buf347, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_474
        del getitem_475
        del getitem_476
        buf349 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf350 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf344, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf349, (2048, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_75, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_75
        buf351 = buf350[0]
        assert_size_stride(buf351, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf351, 16, 'torch.ops.aten.convolution_backward.default')
        del buf350
        buf352 = buf349; del buf349  # reuse
        buf353 = reinterpret_tensor(buf295, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_37.run(buf347, convert_element_type_76, buf353, 411041792, stream=stream0)
        del convert_element_type_76
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf354 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf344, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf353, reinterpret_tensor(buf352, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf355 = buf354[1]
        assert_size_stride(buf355, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf355, 16, 'torch.ops.aten.convolution_backward.default')
        del buf354
        buf356 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf355, buf356, 262144, stream=stream0)
        del buf355
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_473, reinterpret_tensor(buf357, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_473
        buf359 = reinterpret_tensor(buf353, (802816, 512), (512, 1), 0); del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_468, getitem_469, getitem_470, buf359, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_468
        del getitem_469
        del getitem_470
        buf363 = reinterpret_tensor(buf347, (2048, 1024, 196), (200704, 196, 1), 0); del buf347  # reuse
        buf367 = reinterpret_tensor(buf156, (2048, 1024, 196), (200704, 196, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_41.run(buf294, buf351, buf357, buf363, buf367, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf359, (2048, 1024, 196), (200704, 196, 1), 0), buf363, buf361, buf362, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf366 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf359, (2048, 1024, 196), (200704, 196, 1), 0), buf367, rsqrt_36, primals_220, buf361, buf362, buf366, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_220
        del rsqrt_36
        buf369 = reinterpret_tensor(buf344, (200704, 512), (512, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_461, getitem_462, getitem_463, buf369, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_461
        del getitem_462
        del getitem_463
        buf371 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf372 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf366, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf371, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_73, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_73
        buf373 = buf372[0]
        assert_size_stride(buf373, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf373, 16, 'torch.ops.aten.convolution_backward.default')
        del buf372
        buf374 = buf371; del buf371  # reuse
        buf375 = reinterpret_tensor(buf345, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf369, convert_element_type_74, buf375, 102760448, stream=stream0)
        del convert_element_type_74
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf376 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf366, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), buf375, reinterpret_tensor(buf374, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf377 = buf376[1]
        assert_size_stride(buf377, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf377, 16, 'torch.ops.aten.convolution_backward.default')
        del buf376
        buf378 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf377, buf378, 262144, stream=stream0)
        del buf377
        buf379 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf401 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf246, buf379, buf401, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_460, reinterpret_tensor(buf379, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_460
        buf381 = reinterpret_tensor(buf375, (200704, 512), (512, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_455, getitem_456, getitem_457, buf381, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_455
        del getitem_456
        del getitem_457
        buf383 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf384 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf405 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf406 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf251, buf383, buf384, buf405, buf406, 256, stream=stream0)
        buf385 = reinterpret_tensor(buf369, (2048, 256, 196), (50176, 196, 1), 0); del buf369  # reuse
        buf389 = reinterpret_tensor(buf337, (2048, 256, 196), (50176, 196, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf373, buf379, buf385, buf389, 524288, 196, stream=stream0)
        del buf373
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf381, (2048, 256, 196), (50176, 196, 1), 0), buf385, buf383, buf384, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf388 = buf385; del buf385  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf381, (2048, 256, 196), (50176, 196, 1), 0), buf389, rsqrt_35, primals_214, buf383, buf384, buf388, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_214
        del rsqrt_35
        buf391 = reinterpret_tensor(buf389, (200704, 512), (512, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_448, getitem_449, getitem_450, buf391, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_448
        del getitem_449
        del getitem_450
        buf393 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf394 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf388, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf393, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_71, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_71
        buf395 = buf394[0]
        assert_size_stride(buf395, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf395, 16, 'torch.ops.aten.convolution_backward.default')
        del buf394
        buf396 = buf393; del buf393  # reuse
        buf397 = reinterpret_tensor(buf381, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf391, convert_element_type_72, buf397, 102760448, stream=stream0)
        del convert_element_type_72
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf398 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf388, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf397, reinterpret_tensor(buf396, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf399 = buf398[1]
        assert_size_stride(buf399, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf399, 16, 'torch.ops.aten.convolution_backward.default')
        del buf398
        buf400 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_36.run(buf399, buf400, 589824, stream=stream0)
        del buf399
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_447, reinterpret_tensor(buf401, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_447
        buf403 = reinterpret_tensor(buf397, (200704, 512), (512, 1), 0); del buf397  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_442, getitem_443, getitem_444, buf403, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_442
        del getitem_443
        del getitem_444
        buf407 = buf388; del buf388  # reuse
        buf411 = reinterpret_tensor(buf391, (2048, 256, 196), (50176, 196, 1), 0); del buf391  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf395, buf401, buf407, buf411, 524288, 196, stream=stream0)
        del buf395
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf403, (2048, 256, 196), (50176, 196, 1), 0), buf407, buf405, buf406, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf410 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf403, (2048, 256, 196), (50176, 196, 1), 0), buf411, rsqrt_34, primals_208, buf405, buf406, buf410, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_208
        del rsqrt_34
        buf413 = reinterpret_tensor(buf366, (802816, 512), (512, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_435, getitem_436, getitem_437, buf413, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_435
        del getitem_436
        del getitem_437
        buf415 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf416 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf410, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf415, (2048, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_69, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_69
        buf417 = buf416[0]
        assert_size_stride(buf417, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf417, 16, 'torch.ops.aten.convolution_backward.default')
        del buf416
        buf418 = buf415; del buf415  # reuse
        buf419 = reinterpret_tensor(buf367, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf367  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_37.run(buf413, convert_element_type_70, buf419, 411041792, stream=stream0)
        del convert_element_type_70
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf420 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf410, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf419, reinterpret_tensor(buf418, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf421 = buf420[1]
        assert_size_stride(buf421, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf421, 16, 'torch.ops.aten.convolution_backward.default')
        del buf420
        buf422 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf421, buf422, 262144, stream=stream0)
        del buf421
        buf423 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        buf488 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf222, buf423, buf488, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_434, reinterpret_tensor(buf423, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_434
        buf425 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_42.run(buf425, buf351, buf357, buf417, buf423, 2097152, 196, stream=stream0)
        buf426 = reinterpret_tensor(buf417, (802816, 512), (512, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_429, getitem_430, getitem_431, buf426, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_429
        del getitem_430
        del getitem_431
        buf428 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf429 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf492 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf493 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_40.run(buf227, buf428, buf429, buf492, buf493, 1024, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf426, (2048, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf425, (2048, 1024, 196), (200704, 196, 1), 0), buf428, buf429, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf432 = reinterpret_tensor(buf351, (2048, 1024, 196), (200704, 196, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf426, (2048, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf425, (2048, 1024, 196), (200704, 196, 1), 0), rsqrt_33, primals_202, buf428, buf429, buf432, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_202
        del rsqrt_33
        buf434 = reinterpret_tensor(buf410, (200704, 512), (512, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_422, getitem_423, getitem_424, buf434, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_422
        del getitem_423
        del getitem_424
        buf436 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf437 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf432, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf436, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_67, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_67
        buf438 = buf437[0]
        assert_size_stride(buf438, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf438, 16, 'torch.ops.aten.convolution_backward.default')
        del buf437
        buf439 = buf436; del buf436  # reuse
        buf440 = reinterpret_tensor(buf411, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf434, convert_element_type_68, buf440, 102760448, stream=stream0)
        del convert_element_type_68
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf441 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf432, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), buf440, reinterpret_tensor(buf439, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf442 = buf441[1]
        assert_size_stride(buf442, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf442, 16, 'torch.ops.aten.convolution_backward.default')
        del buf441
        buf443 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf442, buf443, 262144, stream=stream0)
        del buf442
        buf444 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf466 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf246, buf444, buf466, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_421, reinterpret_tensor(buf444, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_421
        buf446 = reinterpret_tensor(buf440, (200704, 512), (512, 1), 0); del buf440  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_416, getitem_417, getitem_418, buf446, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_416
        del getitem_417
        del getitem_418
        buf448 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf449 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf470 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf471 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf251, buf448, buf449, buf470, buf471, 256, stream=stream0)
        buf450 = reinterpret_tensor(buf434, (2048, 256, 196), (50176, 196, 1), 0); del buf434  # reuse
        buf454 = reinterpret_tensor(buf403, (2048, 256, 196), (50176, 196, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf438, buf444, buf450, buf454, 524288, 196, stream=stream0)
        del buf438
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf446, (2048, 256, 196), (50176, 196, 1), 0), buf450, buf448, buf449, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf453 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf446, (2048, 256, 196), (50176, 196, 1), 0), buf454, rsqrt_32, primals_196, buf448, buf449, buf453, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_196
        del rsqrt_32
        buf456 = reinterpret_tensor(buf454, (200704, 512), (512, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_409, getitem_410, getitem_411, buf456, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_409
        del getitem_410
        del getitem_411
        buf458 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf459 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf453, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf458, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_65, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_65
        buf460 = buf459[0]
        assert_size_stride(buf460, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf460, 16, 'torch.ops.aten.convolution_backward.default')
        del buf459
        buf461 = buf458; del buf458  # reuse
        buf462 = reinterpret_tensor(buf446, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf446  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf456, convert_element_type_66, buf462, 102760448, stream=stream0)
        del convert_element_type_66
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf463 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf453, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf462, reinterpret_tensor(buf461, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf464 = buf463[1]
        assert_size_stride(buf464, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf464, 16, 'torch.ops.aten.convolution_backward.default')
        del buf463
        buf465 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_36.run(buf464, buf465, 589824, stream=stream0)
        del buf464
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_408, reinterpret_tensor(buf466, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_408
        buf468 = reinterpret_tensor(buf462, (200704, 512), (512, 1), 0); del buf462  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_403, getitem_404, getitem_405, buf468, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_403
        del getitem_404
        del getitem_405
        buf472 = buf453; del buf453  # reuse
        buf476 = reinterpret_tensor(buf456, (2048, 256, 196), (50176, 196, 1), 0); del buf456  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf460, buf466, buf472, buf476, 524288, 196, stream=stream0)
        del buf460
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf468, (2048, 256, 196), (50176, 196, 1), 0), buf472, buf470, buf471, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf475 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf468, (2048, 256, 196), (50176, 196, 1), 0), buf476, rsqrt_31, primals_190, buf470, buf471, buf475, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_190
        del rsqrt_31
        buf478 = reinterpret_tensor(buf432, (802816, 512), (512, 1), 0); del buf432  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_396, getitem_397, getitem_398, buf478, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_396
        del getitem_397
        del getitem_398
        buf480 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf481 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf475, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf480, (2048, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_63, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_63
        buf482 = buf481[0]
        assert_size_stride(buf482, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf482, 16, 'torch.ops.aten.convolution_backward.default')
        del buf481
        buf483 = buf480; del buf480  # reuse
        buf484 = reinterpret_tensor(buf426, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_37.run(buf478, convert_element_type_64, buf484, 411041792, stream=stream0)
        del convert_element_type_64
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf485 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf475, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf484, reinterpret_tensor(buf483, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf486 = buf485[1]
        assert_size_stride(buf486, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf486, 16, 'torch.ops.aten.convolution_backward.default')
        del buf485
        buf487 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf486, buf487, 262144, stream=stream0)
        del buf486
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_395, reinterpret_tensor(buf488, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_395
        buf490 = reinterpret_tensor(buf484, (802816, 512), (512, 1), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_390, getitem_391, getitem_392, buf490, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_390
        del getitem_391
        del getitem_392
        buf494 = reinterpret_tensor(buf478, (2048, 1024, 196), (200704, 196, 1), 0); del buf478  # reuse
        buf498 = reinterpret_tensor(buf419, (2048, 1024, 196), (200704, 196, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_41.run(buf425, buf482, buf488, buf494, buf498, 2097152, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf490, (2048, 1024, 196), (200704, 196, 1), 0), buf494, buf492, buf493, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf497 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf490, (2048, 1024, 196), (200704, 196, 1), 0), buf498, rsqrt_30, primals_184, buf492, buf493, buf497, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_184
        del rsqrt_30
        buf500 = reinterpret_tensor(buf475, (200704, 512), (512, 1), 0); del buf475  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_383, getitem_384, getitem_385, buf500, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_383
        del getitem_384
        del getitem_385
        buf502 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf503 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf497, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf502, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_61, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_61
        buf504 = buf503[0]
        assert_size_stride(buf504, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf504, 16, 'torch.ops.aten.convolution_backward.default')
        del buf503
        buf505 = buf502; del buf502  # reuse
        buf506 = reinterpret_tensor(buf476, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf500, convert_element_type_62, buf506, 102760448, stream=stream0)
        del convert_element_type_62
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf507 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf497, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), buf506, reinterpret_tensor(buf505, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf508 = buf507[1]
        assert_size_stride(buf508, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf508, 16, 'torch.ops.aten.convolution_backward.default')
        del buf507
        buf509 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf508, buf509, 262144, stream=stream0)
        del buf508
        buf510 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf532 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf246, buf510, buf532, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_382, reinterpret_tensor(buf510, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_382
        buf512 = reinterpret_tensor(buf506, (200704, 512), (512, 1), 0); del buf506  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_377, getitem_378, getitem_379, buf512, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_377
        del getitem_378
        del getitem_379
        buf514 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf515 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf536 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf537 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf251, buf514, buf515, buf536, buf537, 256, stream=stream0)
        buf516 = reinterpret_tensor(buf500, (2048, 256, 196), (50176, 196, 1), 0); del buf500  # reuse
        buf520 = reinterpret_tensor(buf468, (2048, 256, 196), (50176, 196, 1), 0); del buf468  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf504, buf510, buf516, buf520, 524288, 196, stream=stream0)
        del buf504
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf512, (2048, 256, 196), (50176, 196, 1), 0), buf516, buf514, buf515, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf519 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf512, (2048, 256, 196), (50176, 196, 1), 0), buf520, rsqrt_29, primals_178, buf514, buf515, buf519, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_178
        del rsqrt_29
        buf522 = reinterpret_tensor(buf520, (200704, 512), (512, 1), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_370, getitem_371, getitem_372, buf522, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_370
        del getitem_371
        del getitem_372
        buf524 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf525 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf519, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf524, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_59, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_59
        buf526 = buf525[0]
        assert_size_stride(buf526, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf526, 16, 'torch.ops.aten.convolution_backward.default')
        del buf525
        buf527 = buf524; del buf524  # reuse
        buf528 = reinterpret_tensor(buf512, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf512  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf522, convert_element_type_60, buf528, 102760448, stream=stream0)
        del convert_element_type_60
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf529 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf519, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf528, reinterpret_tensor(buf527, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf530 = buf529[1]
        assert_size_stride(buf530, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf530, 16, 'torch.ops.aten.convolution_backward.default')
        del buf529
        buf531 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_36.run(buf530, buf531, 589824, stream=stream0)
        del buf530
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_369, reinterpret_tensor(buf532, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_369
        buf534 = reinterpret_tensor(buf528, (200704, 512), (512, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_364, getitem_365, getitem_366, buf534, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_364
        del getitem_365
        del getitem_366
        buf538 = buf519; del buf519  # reuse
        buf542 = reinterpret_tensor(buf522, (2048, 256, 196), (50176, 196, 1), 0); del buf522  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf526, buf532, buf538, buf542, 524288, 196, stream=stream0)
        del buf526
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf534, (2048, 256, 196), (50176, 196, 1), 0), buf538, buf536, buf537, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf541 = buf538; del buf538  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf534, (2048, 256, 196), (50176, 196, 1), 0), buf542, rsqrt_28, primals_172, buf536, buf537, buf541, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del primals_172
        del rsqrt_28
        buf544 = reinterpret_tensor(buf497, (802816, 512), (512, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_357, getitem_358, getitem_359, buf544, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_357
        del getitem_358
        del getitem_359
        buf546 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf547 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf541, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf546, (2048, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_57, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_57
        buf548 = buf547[0]
        assert_size_stride(buf548, (2048, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf548, 16, 'torch.ops.aten.convolution_backward.default')
        del buf547
        buf549 = buf546; del buf546  # reuse
        buf550 = reinterpret_tensor(buf498, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_37.run(buf544, convert_element_type_58, buf550, 411041792, stream=stream0)
        del convert_element_type_58
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf551 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf541, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf550, reinterpret_tensor(buf549, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf552 = buf551[1]
        assert_size_stride(buf552, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf552, 16, 'torch.ops.aten.convolution_backward.default')
        del buf551
        buf553 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf552, buf553, 262144, stream=stream0)
        del buf552
        buf554 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        buf617 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf222, buf554, buf617, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_356, reinterpret_tensor(buf554, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_356
        buf556 = reinterpret_tensor(buf550, (802816, 512), (512, 1), 0); del buf550  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_351, getitem_352, getitem_353, buf556, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_351
        del getitem_352
        del getitem_353
        buf558 = buf482; del buf482  # reuse
        buf561 = reinterpret_tensor(buf544, (2048, 1024, 196), (200704, 196, 1), 0); del buf544  # reuse
        buf565 = reinterpret_tensor(buf490, (2048, 1024, 196), (200704, 196, 1), 0); del buf490  # reuse
        buf580 = reinterpret_tensor(buf413, (2048, 1024, 196), (200704, 196, 1), 0); del buf413  # reuse
        buf584 = reinterpret_tensor(buf359, (2048, 1024, 196), (200704, 196, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_43.run(buf558, buf425, buf488, buf548, buf554, buf561, buf565, buf580, buf584, 2097152, 196, stream=stream0)
        del buf425
        del buf548
        del buf558
        buf559 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf560 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf579 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_44.run(buf227, buf559, buf560, buf579, 1024, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf556, (2048, 1024, 196), (200704, 196, 1), 0), buf561, buf559, buf560, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf564 = buf561; del buf561  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf556, (2048, 1024, 196), (200704, 196, 1), 0), buf565, rsqrt_27, primals_166, buf559, buf560, buf564, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del buf556
        del buf565
        del primals_166
        del rsqrt_27
        buf567 = empty_strided_cuda((1605632, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_344, getitem_345, getitem_346, buf567, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_344
        del getitem_345
        del getitem_346
        buf569 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf570 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf564, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf569, (2048, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_55, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_55
        buf571 = buf570[0]
        assert_size_stride(buf571, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf571, 16, 'torch.ops.aten.convolution_backward.default')
        del buf570
        buf572 = buf569; del buf569  # reuse
        buf629 = empty_strided_cuda((1605632, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_308, getitem_309, getitem_310, buf629, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_308
        del getitem_309
        del getitem_310
        buf573 = empty_strided_cuda((2048, 512, 28, 28), (401408, 784, 28, 1), torch.bfloat16)
        buf635 = empty_strided_cuda((2048, 512, 28, 28), (401408, 784, 28, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_45.run(buf567, convert_element_type_50, buf629, buf573, buf635, 822083584, stream=stream0)
        del buf567
        del convert_element_type_50
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf574 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf564, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), buf573, reinterpret_tensor(buf572, (1024, 512, 1, 1), (0, 0, 0, 0), 0), None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf575 = buf574[1]
        assert_size_stride(buf575, (1024, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf575, 16, 'torch.ops.aten.convolution_backward.default')
        del buf574
        buf576 = empty_strided_cuda((1024, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_23.run(buf575, buf576, 524288, stream=stream0)
        del buf575
        buf577 = reinterpret_tensor(buf564, (802816, 512), (512, 1), 0); del buf564  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_341, getitem_342, getitem_343, buf577, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_341
        del getitem_342
        del getitem_343
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf577, (2048, 1024, 196), (200704, 196, 1), 0), buf580, buf227, buf579, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        buf583 = buf580; del buf580  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf577, (2048, 1024, 196), (200704, 196, 1), 0), buf584, rsqrt_26, primals_160, buf227, buf579, buf583, 401408, 196, 200704, 196, 1024, 1024, 392, 1, stream=stream0)
        del primals_160
        del rsqrt_26
        buf586 = reinterpret_tensor(buf541, (200704, 512), (512, 1), 0); del buf541  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_334, getitem_335, getitem_336, buf586, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_334
        del getitem_335
        del getitem_336
        buf588 = buf572; del buf572  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf589 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf583, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf588, (2048, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_53
        buf590 = buf589[0]
        assert_size_stride(buf590, (2048, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf590, 16, 'torch.ops.aten.convolution_backward.default')
        del buf589
        buf591 = buf588; del buf588  # reuse
        buf592 = reinterpret_tensor(buf542, (2048, 256, 14, 14), (50176, 196, 14, 1), 0); del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_29.run(buf586, convert_element_type_54, buf592, 102760448, stream=stream0)
        del convert_element_type_54
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf593 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf583, (2048, 1024, 14, 14), (200704, 196, 14, 1), 0), buf592, reinterpret_tensor(buf591, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf594 = buf593[1]
        assert_size_stride(buf594, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf594, 16, 'torch.ops.aten.convolution_backward.default')
        del buf593
        buf595 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf594, buf595, 262144, stream=stream0)
        del buf594
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_333, buf246, 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del buf247
        del buf270
        del buf313
        del buf335
        del buf379
        del buf401
        del buf444
        del buf466
        del buf510
        del buf532
        del getitem_333
        buf597 = reinterpret_tensor(buf592, (200704, 512), (512, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_328, getitem_329, getitem_330, buf597, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_328
        del getitem_329
        del getitem_330
        buf599 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf600 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf621 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf622 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf251, buf599, buf600, buf621, buf622, 256, stream=stream0)
        buf601 = reinterpret_tensor(buf586, (2048, 256, 196), (50176, 196, 1), 0); del buf586  # reuse
        buf605 = reinterpret_tensor(buf534, (2048, 256, 196), (50176, 196, 1), 0); del buf534  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf590, buf246, buf601, buf605, 524288, 196, stream=stream0)
        del buf246
        del buf590
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf597, (2048, 256, 196), (50176, 196, 1), 0), buf601, buf599, buf600, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        buf604 = buf601; del buf601  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf597, (2048, 256, 196), (50176, 196, 1), 0), buf605, rsqrt_25, primals_154, buf599, buf600, buf604, 401408, 196, 50176, 196, 1024, 256, 392, 1, stream=stream0)
        del buf597
        del buf605
        del primals_154
        del rsqrt_25
        buf607 = reinterpret_tensor(buf583, (802816, 512), (512, 1), 0); del buf583  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_321, getitem_322, getitem_323, buf607, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_321
        del getitem_322
        del getitem_323
        buf609 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf610 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf604, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf609, (2048, 256, 28, 28), (0, 0, 0, 0), 0), convert_element_type_51, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_51
        buf611 = buf610[0]
        assert_size_stride(buf611, (2048, 256, 28, 28), (200704, 1, 7168, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf611, 16, 'torch.ops.aten.convolution_backward.default')
        del buf610
        buf612 = buf609; del buf609  # reuse
        buf613 = reinterpret_tensor(buf584, (2048, 256, 28, 28), (200704, 784, 28, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_46.run(buf607, convert_element_type_52, buf613, 411041792, stream=stream0)
        del convert_element_type_52
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf614 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf604, (2048, 256, 14, 14), (50176, 196, 14, 1), 0), buf613, reinterpret_tensor(buf612, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        del buf604
        buf615 = buf614[1]
        assert_size_stride(buf615, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf615, 16, 'torch.ops.aten.convolution_backward.default')
        del buf614
        buf616 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_36.run(buf615, buf616, 589824, stream=stream0)
        del buf615
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_320, reinterpret_tensor(buf617, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_320
        buf619 = reinterpret_tensor(buf613, (802816, 512), (512, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_315, getitem_316, getitem_317, buf619, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_315
        del getitem_316
        del getitem_317
        buf623 = reinterpret_tensor(buf607, (2048, 256, 784), (200704, 784, 1), 0); del buf607  # reuse
        buf627 = reinterpret_tensor(buf577, (2048, 256, 784), (200704, 784, 1), 0); del buf577  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf611, buf617, buf623, buf627, 524288, 784, stream=stream0)
        del buf611
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_12.run(reinterpret_tensor(buf619, (2048, 256, 784), (200704, 784, 1), 0), buf623, buf621, buf622, 1605632, 784, 200704, 784, 1024, 256, 1568, 1, stream=stream0)
        buf626 = buf623; del buf623  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_13.run(reinterpret_tensor(buf619, (2048, 256, 784), (200704, 784, 1), 0), buf627, rsqrt_24, primals_148, buf621, buf622, buf626, 1605632, 784, 200704, 784, 1024, 256, 1568, 1, stream=stream0)
        del primals_148
        del rsqrt_24
        buf631 = buf612; del buf612  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf632 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf626, (2048, 256, 28, 28), (200704, 784, 28, 1), 0), reinterpret_tensor(buf631, (2048, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_49, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_49
        buf633 = buf632[0]
        assert_size_stride(buf633, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf633, 16, 'torch.ops.aten.convolution_backward.default')
        del buf632
        buf634 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf636 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf626, (2048, 256, 28, 28), (200704, 784, 28, 1), 0), buf635, reinterpret_tensor(buf634, (256, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf637 = buf636[1]
        assert_size_stride(buf637, (256, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf637, 16, 'torch.ops.aten.convolution_backward.default')
        del buf636
        buf638 = empty_strided_cuda((256, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_48.run(buf637, buf638, 131072, stream=stream0)
        del buf637
        buf639 = empty_strided_cuda((1605632, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_342], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_49.run(buf639, 822083584, stream=stream0)
        buf640 = empty_strided_cuda((822083584, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_50.run(buf639, buf640, 822083584, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_307, reinterpret_tensor(buf640, (1605632, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        del getitem_307
        buf642 = reinterpret_tensor(buf635, (1605632, 512), (512, 1), 0); del buf635  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_302, getitem_303, getitem_304, buf642, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_302
        del getitem_303
        del getitem_304
        buf644 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf645 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_51.run(buf34, buf644, buf645, 512, stream=stream0)
        buf646 = reinterpret_tensor(buf573, (2048, 512, 784), (401408, 784, 1), 0); del buf573  # reuse
        buf650 = reinterpret_tensor(buf629, (2048, 512, 784), (401408, 784, 1), 0); del buf629  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_52.run(buf571, buf633, buf640, buf646, buf650, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf642, (2048, 512, 784), (401408, 784, 1), 0), buf646, buf644, buf645, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf649 = buf646; del buf646  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf642, (2048, 512, 784), (401408, 784, 1), 0), buf650, rsqrt_23, primals_142, buf644, buf645, buf649, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del primals_142
        del rsqrt_23
        buf652 = reinterpret_tensor(buf209, (401408, 512), (512, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_295, getitem_296, getitem_297, buf652, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_295
        del getitem_296
        del getitem_297
        buf654 = buf634; del buf634  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf655 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf649, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf654, (2048, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_47, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_47
        buf656 = buf655[0]
        assert_size_stride(buf656, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf656, 16, 'torch.ops.aten.convolution_backward.default')
        del buf655
        buf657 = buf654; del buf654  # reuse
        buf658 = reinterpret_tensor(buf210, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53.run(buf652, convert_element_type_48, buf658, 205520896, stream=stream0)
        del convert_element_type_48
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf659 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf649, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), buf658, reinterpret_tensor(buf657, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf660 = buf659[1]
        assert_size_stride(buf660, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf660, 16, 'torch.ops.aten.convolution_backward.default')
        del buf659
        buf661 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(buf660, buf661, 65536, stream=stream0)
        del buf660
        buf662 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        buf685 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(buf5, buf662, buf685, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_294, reinterpret_tensor(buf662, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_294
        buf664 = reinterpret_tensor(buf658, (401408, 512), (512, 1), 0); del buf658  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_289, getitem_290, getitem_291, buf664, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_289
        del getitem_290
        del getitem_291
        buf666 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_55.run(buf666, 128, stream=stream0)
        buf667 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf668 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf689 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf690 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_56.run(buf666, buf667, buf668, buf689, buf690, 128, stream=stream0)
        buf669 = reinterpret_tensor(buf652, (2048, 128, 784), (100352, 784, 1), 0); del buf652  # reuse
        buf673 = reinterpret_tensor(buf202, (2048, 128, 784), (100352, 784, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf656, buf662, buf669, buf673, 262144, 784, stream=stream0)
        del buf656
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf664, (2048, 128, 784), (100352, 784, 1), 0), buf669, buf667, buf668, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf672 = buf669; del buf669  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf664, (2048, 128, 784), (100352, 784, 1), 0), buf673, rsqrt_22, primals_136, buf667, buf668, buf672, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del primals_136
        del rsqrt_22
        buf675 = reinterpret_tensor(buf673, (401408, 512), (512, 1), 0); del buf673  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_282, getitem_283, getitem_284, buf675, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_282
        del getitem_283
        del getitem_284
        buf677 = buf657; del buf657  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf678 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf672, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf677, (2048, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_45, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_45
        buf679 = buf678[0]
        assert_size_stride(buf679, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf679, 16, 'torch.ops.aten.convolution_backward.default')
        del buf678
        buf680 = buf677; del buf677  # reuse
        buf681 = reinterpret_tensor(buf664, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf664  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53.run(buf675, convert_element_type_46, buf681, 205520896, stream=stream0)
        del convert_element_type_46
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf682 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf672, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf681, reinterpret_tensor(buf680, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf683 = buf682[1]
        assert_size_stride(buf683, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf683, 16, 'torch.ops.aten.convolution_backward.default')
        del buf682
        buf684 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_58.run(buf683, buf684, 147456, stream=stream0)
        del buf683
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_281, reinterpret_tensor(buf685, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_281
        buf687 = reinterpret_tensor(buf681, (401408, 512), (512, 1), 0); del buf681  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_276, getitem_277, getitem_278, buf687, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_276
        del getitem_277
        del getitem_278
        buf691 = buf672; del buf672  # reuse
        buf695 = reinterpret_tensor(buf675, (2048, 128, 784), (100352, 784, 1), 0); del buf675  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf679, buf685, buf691, buf695, 262144, 784, stream=stream0)
        del buf679
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf687, (2048, 128, 784), (100352, 784, 1), 0), buf691, buf689, buf690, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf694 = buf691; del buf691  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf687, (2048, 128, 784), (100352, 784, 1), 0), buf695, rsqrt_21, primals_130, buf689, buf690, buf694, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del primals_130
        del rsqrt_21
        buf697 = reinterpret_tensor(buf649, (1605632, 512), (512, 1), 0); del buf649  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_269, getitem_270, getitem_271, buf697, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_269
        del getitem_270
        del getitem_271
        buf699 = buf680; del buf680  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf700 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf694, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf699, (2048, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_43, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_43
        buf701 = buf700[0]
        assert_size_stride(buf701, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf701, 16, 'torch.ops.aten.convolution_backward.default')
        del buf700
        buf702 = buf699; del buf699  # reuse
        buf703 = reinterpret_tensor(buf650, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_59.run(buf697, convert_element_type_44, buf703, 822083584, stream=stream0)
        del convert_element_type_44
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf704 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf694, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf703, reinterpret_tensor(buf702, (128, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf705 = buf704[1]
        assert_size_stride(buf705, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf705, 16, 'torch.ops.aten.convolution_backward.default')
        del buf704
        buf706 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(buf705, buf706, 65536, stream=stream0)
        del buf705
        buf707 = empty_strided_cuda((822083584, ), (1, ), torch.int8)
        buf772 = empty_strided_cuda((822083584, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_60.run(buf639, buf707, buf772, 822083584, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_268, reinterpret_tensor(buf707, (1605632, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        del getitem_268
        buf709 = buf703; del buf703  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_61.run(buf571, buf633, buf640, buf701, buf707, buf709, 1048576, 784, stream=stream0)
        buf710 = reinterpret_tensor(buf701, (1605632, 512), (512, 1), 0); del buf701  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_263, getitem_264, getitem_265, buf710, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_263
        del getitem_264
        del getitem_265
        buf712 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf713 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf776 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf777 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf34, buf712, buf713, buf776, buf777, 512, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf710, (2048, 512, 784), (401408, 784, 1), 0), reinterpret_tensor(buf709, (2048, 512, 784), (401408, 784, 1), 0), buf712, buf713, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf716 = reinterpret_tensor(buf633, (2048, 512, 784), (401408, 784, 1), 0); del buf633  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf710, (2048, 512, 784), (401408, 784, 1), 0), reinterpret_tensor(buf709, (2048, 512, 784), (401408, 784, 1), 0), rsqrt_20, primals_124, buf712, buf713, buf716, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del primals_124
        del rsqrt_20
        buf718 = reinterpret_tensor(buf694, (401408, 512), (512, 1), 0); del buf694  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_256, getitem_257, getitem_258, buf718, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_256
        del getitem_257
        del getitem_258
        buf720 = buf702; del buf702  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf721 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf716, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf720, (2048, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_41
        buf722 = buf721[0]
        assert_size_stride(buf722, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf722, 16, 'torch.ops.aten.convolution_backward.default')
        del buf721
        buf723 = buf720; del buf720  # reuse
        buf724 = reinterpret_tensor(buf695, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf695  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53.run(buf718, convert_element_type_42, buf724, 205520896, stream=stream0)
        del convert_element_type_42
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf725 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf716, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), buf724, reinterpret_tensor(buf723, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf726 = buf725[1]
        assert_size_stride(buf726, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf726, 16, 'torch.ops.aten.convolution_backward.default')
        del buf725
        buf727 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(buf726, buf727, 65536, stream=stream0)
        del buf726
        buf728 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        buf750 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(buf5, buf728, buf750, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_255, reinterpret_tensor(buf728, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_255
        buf730 = reinterpret_tensor(buf724, (401408, 512), (512, 1), 0); del buf724  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_250, getitem_251, getitem_252, buf730, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_250
        del getitem_251
        del getitem_252
        buf732 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf733 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf754 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf755 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_56.run(buf666, buf732, buf733, buf754, buf755, 128, stream=stream0)
        buf734 = reinterpret_tensor(buf718, (2048, 128, 784), (100352, 784, 1), 0); del buf718  # reuse
        buf738 = reinterpret_tensor(buf687, (2048, 128, 784), (100352, 784, 1), 0); del buf687  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf722, buf728, buf734, buf738, 262144, 784, stream=stream0)
        del buf722
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf730, (2048, 128, 784), (100352, 784, 1), 0), buf734, buf732, buf733, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf737 = buf734; del buf734  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf730, (2048, 128, 784), (100352, 784, 1), 0), buf738, rsqrt_19, primals_118, buf732, buf733, buf737, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del primals_118
        del rsqrt_19
        buf740 = reinterpret_tensor(buf738, (401408, 512), (512, 1), 0); del buf738  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_243, getitem_244, getitem_245, buf740, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_243
        del getitem_244
        del getitem_245
        buf742 = buf723; del buf723  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf743 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf737, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf742, (2048, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_39
        buf744 = buf743[0]
        assert_size_stride(buf744, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf744, 16, 'torch.ops.aten.convolution_backward.default')
        del buf743
        buf745 = buf742; del buf742  # reuse
        buf746 = reinterpret_tensor(buf730, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf730  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53.run(buf740, convert_element_type_40, buf746, 205520896, stream=stream0)
        del convert_element_type_40
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf747 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf737, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf746, reinterpret_tensor(buf745, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf748 = buf747[1]
        assert_size_stride(buf748, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf748, 16, 'torch.ops.aten.convolution_backward.default')
        del buf747
        buf749 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_58.run(buf748, buf749, 147456, stream=stream0)
        del buf748
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_242, reinterpret_tensor(buf750, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_242
        buf752 = reinterpret_tensor(buf746, (401408, 512), (512, 1), 0); del buf746  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_237, getitem_238, getitem_239, buf752, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_237
        del getitem_238
        del getitem_239
        buf756 = buf737; del buf737  # reuse
        buf760 = reinterpret_tensor(buf740, (2048, 128, 784), (100352, 784, 1), 0); del buf740  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf744, buf750, buf756, buf760, 262144, 784, stream=stream0)
        del buf744
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf752, (2048, 128, 784), (100352, 784, 1), 0), buf756, buf754, buf755, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf759 = buf756; del buf756  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf752, (2048, 128, 784), (100352, 784, 1), 0), buf760, rsqrt_18, primals_112, buf754, buf755, buf759, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del primals_112
        del rsqrt_18
        buf762 = reinterpret_tensor(buf716, (1605632, 512), (512, 1), 0); del buf716  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_230, getitem_231, getitem_232, buf762, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_230
        del getitem_231
        del getitem_232
        buf764 = buf745; del buf745  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf765 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf759, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf764, (2048, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_37
        buf766 = buf765[0]
        assert_size_stride(buf766, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf766, 16, 'torch.ops.aten.convolution_backward.default')
        del buf765
        buf767 = buf764; del buf764  # reuse
        buf768 = reinterpret_tensor(buf710, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf710  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_59.run(buf762, convert_element_type_38, buf768, 822083584, stream=stream0)
        del convert_element_type_38
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf769 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf759, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf768, reinterpret_tensor(buf767, (128, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf770 = buf769[1]
        assert_size_stride(buf770, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf770, 16, 'torch.ops.aten.convolution_backward.default')
        del buf769
        buf771 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(buf770, buf771, 65536, stream=stream0)
        del buf770
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_229, reinterpret_tensor(buf772, (1605632, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        del getitem_229
        buf774 = reinterpret_tensor(buf768, (1605632, 512), (512, 1), 0); del buf768  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_224, getitem_225, getitem_226, buf774, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_224
        del getitem_225
        del getitem_226
        buf778 = reinterpret_tensor(buf762, (2048, 512, 784), (401408, 784, 1), 0); del buf762  # reuse
        buf782 = reinterpret_tensor(buf571, (2048, 512, 784), (401408, 784, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_62.run(buf709, buf766, buf772, buf778, buf782, 1048576, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf774, (2048, 512, 784), (401408, 784, 1), 0), buf778, buf776, buf777, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf781 = buf778; del buf778  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf774, (2048, 512, 784), (401408, 784, 1), 0), buf782, rsqrt_17, primals_106, buf776, buf777, buf781, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del primals_106
        del rsqrt_17
        buf784 = reinterpret_tensor(buf759, (401408, 512), (512, 1), 0); del buf759  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_217, getitem_218, getitem_219, buf784, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_217
        del getitem_218
        del getitem_219
        buf786 = buf767; del buf767  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf787 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf781, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf786, (2048, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_35, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_35
        buf788 = buf787[0]
        assert_size_stride(buf788, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf788, 16, 'torch.ops.aten.convolution_backward.default')
        del buf787
        buf789 = buf786; del buf786  # reuse
        buf790 = reinterpret_tensor(buf760, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf760  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53.run(buf784, convert_element_type_36, buf790, 205520896, stream=stream0)
        del convert_element_type_36
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf791 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf781, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), buf790, reinterpret_tensor(buf789, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf792 = buf791[1]
        assert_size_stride(buf792, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf792, 16, 'torch.ops.aten.convolution_backward.default')
        del buf791
        buf793 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(buf792, buf793, 65536, stream=stream0)
        del buf792
        buf794 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        buf816 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(buf5, buf794, buf816, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_216, reinterpret_tensor(buf794, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_216
        buf796 = reinterpret_tensor(buf790, (401408, 512), (512, 1), 0); del buf790  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_211, getitem_212, getitem_213, buf796, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_211
        del getitem_212
        del getitem_213
        buf798 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf799 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf820 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf821 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_56.run(buf666, buf798, buf799, buf820, buf821, 128, stream=stream0)
        buf800 = reinterpret_tensor(buf784, (2048, 128, 784), (100352, 784, 1), 0); del buf784  # reuse
        buf804 = reinterpret_tensor(buf752, (2048, 128, 784), (100352, 784, 1), 0); del buf752  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf788, buf794, buf800, buf804, 262144, 784, stream=stream0)
        del buf788
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf796, (2048, 128, 784), (100352, 784, 1), 0), buf800, buf798, buf799, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf803 = buf800; del buf800  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf796, (2048, 128, 784), (100352, 784, 1), 0), buf804, rsqrt_16, primals_100, buf798, buf799, buf803, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del primals_100
        del rsqrt_16
        buf806 = reinterpret_tensor(buf804, (401408, 512), (512, 1), 0); del buf804  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_204, getitem_205, getitem_206, buf806, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_204
        del getitem_205
        del getitem_206
        buf808 = buf789; del buf789  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf809 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf803, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf808, (2048, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_33
        buf810 = buf809[0]
        assert_size_stride(buf810, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf810, 16, 'torch.ops.aten.convolution_backward.default')
        del buf809
        buf811 = buf808; del buf808  # reuse
        buf812 = reinterpret_tensor(buf796, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf796  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53.run(buf806, convert_element_type_34, buf812, 205520896, stream=stream0)
        del convert_element_type_34
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf813 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf803, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf812, reinterpret_tensor(buf811, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf814 = buf813[1]
        assert_size_stride(buf814, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf814, 16, 'torch.ops.aten.convolution_backward.default')
        del buf813
        buf815 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_58.run(buf814, buf815, 147456, stream=stream0)
        del buf814
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_203, reinterpret_tensor(buf816, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_203
        buf818 = reinterpret_tensor(buf812, (401408, 512), (512, 1), 0); del buf812  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_198, getitem_199, getitem_200, buf818, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_198
        del getitem_199
        del getitem_200
        buf822 = buf803; del buf803  # reuse
        buf826 = reinterpret_tensor(buf806, (2048, 128, 784), (100352, 784, 1), 0); del buf806  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf810, buf816, buf822, buf826, 262144, 784, stream=stream0)
        del buf810
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf818, (2048, 128, 784), (100352, 784, 1), 0), buf822, buf820, buf821, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf825 = buf822; del buf822  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf818, (2048, 128, 784), (100352, 784, 1), 0), buf826, rsqrt_15, primals_94, buf820, buf821, buf825, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del primals_94
        del rsqrt_15
        buf828 = reinterpret_tensor(buf781, (1605632, 512), (512, 1), 0); del buf781  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_191, getitem_192, getitem_193, buf828, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_191
        del getitem_192
        del getitem_193
        buf830 = buf811; del buf811  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf831 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf825, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf830, (2048, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_31
        buf832 = buf831[0]
        assert_size_stride(buf832, (2048, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf832, 16, 'torch.ops.aten.convolution_backward.default')
        del buf831
        buf833 = buf830; del buf830  # reuse
        buf834 = reinterpret_tensor(buf782, (2048, 512, 28, 28), (401408, 784, 28, 1), 0); del buf782  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_59.run(buf828, convert_element_type_32, buf834, 822083584, stream=stream0)
        del convert_element_type_32
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf835 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf825, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf834, reinterpret_tensor(buf833, (128, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf836 = buf835[1]
        assert_size_stride(buf836, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf836, 16, 'torch.ops.aten.convolution_backward.default')
        del buf835
        buf837 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(buf836, buf837, 65536, stream=stream0)
        del buf836
        buf838 = empty_strided_cuda((822083584, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_50.run(buf639, buf838, 822083584, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_190, reinterpret_tensor(buf838, (1605632, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        del getitem_190
        buf840 = reinterpret_tensor(buf834, (1605632, 512), (512, 1), 0); del buf834  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_185, getitem_186, getitem_187, buf840, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_185
        del getitem_186
        del getitem_187
        buf842 = buf766; del buf766  # reuse
        buf845 = reinterpret_tensor(buf828, (2048, 512, 784), (401408, 784, 1), 0); del buf828  # reuse
        buf849 = reinterpret_tensor(buf774, (2048, 512, 784), (401408, 784, 1), 0); del buf774  # reuse
        buf864 = reinterpret_tensor(buf697, (2048, 512, 784), (401408, 784, 1), 0); del buf697  # reuse
        buf868 = reinterpret_tensor(buf642, (2048, 512, 784), (401408, 784, 1), 0); del buf642  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_63.run(buf842, buf709, buf772, buf832, buf838, buf845, buf849, buf864, buf868, 1048576, 784, stream=stream0)
        del buf709
        del buf832
        del buf842
        buf843 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf844 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf863 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_64.run(buf34, buf843, buf844, buf863, 512, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf840, (2048, 512, 784), (401408, 784, 1), 0), buf845, buf843, buf844, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf848 = buf845; del buf845  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf840, (2048, 512, 784), (401408, 784, 1), 0), buf849, rsqrt_14, primals_88, buf843, buf844, buf848, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del buf840
        del buf849
        del primals_88
        del rsqrt_14
        buf851 = empty_strided_cuda((3211264, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_178, getitem_179, getitem_180, buf851, 32, 1, 512, 1, 2, 16, 32, 3211264, 1, 1, stream=stream0)
        del getitem_178
        del getitem_179
        del getitem_180
        buf853 = buf833; del buf833  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf854 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf848, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf853, (2048, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_29, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_29
        buf855 = buf854[0]
        assert_size_stride(buf855, (2048, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf855, 16, 'torch.ops.aten.convolution_backward.default')
        del buf854
        buf856 = buf853; del buf853  # reuse
        buf911 = empty_strided_cuda((3211264, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_142, getitem_143, getitem_144, buf911, 32, 1, 512, 1, 2, 16, 32, 3211264, 1, 1, stream=stream0)
        del getitem_142
        del getitem_143
        del getitem_144
        buf857 = empty_strided_cuda((2048, 256, 56, 56), (802816, 3136, 56, 1), torch.bfloat16)
        buf917 = empty_strided_cuda((2048, 256, 56, 56), (802816, 3136, 56, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_65.run(buf851, convert_element_type_24, buf911, buf857, buf917, 1644167168, stream=stream0)
        del buf851
        del convert_element_type_24
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf858 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf848, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), buf857, reinterpret_tensor(buf856, (512, 256, 1, 1), (0, 0, 0, 0), 0), None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf859 = buf858[1]
        assert_size_stride(buf859, (512, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf859, 16, 'torch.ops.aten.convolution_backward.default')
        del buf858
        buf860 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_48.run(buf859, buf860, 131072, stream=stream0)
        del buf859
        buf861 = reinterpret_tensor(buf848, (1605632, 512), (512, 1), 0); del buf848  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_175, getitem_176, getitem_177, buf861, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_175
        del getitem_176
        del getitem_177
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf861, (2048, 512, 784), (401408, 784, 1), 0), buf864, buf34, buf863, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        buf867 = buf864; del buf864  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf861, (2048, 512, 784), (401408, 784, 1), 0), buf868, rsqrt_13, primals_82, buf34, buf863, buf867, 1605632, 784, 401408, 784, 1024, 512, 1568, 1, stream=stream0)
        del primals_82
        del rsqrt_13
        buf870 = reinterpret_tensor(buf825, (401408, 512), (512, 1), 0); del buf825  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_168, getitem_169, getitem_170, buf870, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_168
        del getitem_169
        del getitem_170
        buf872 = buf856; del buf856  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf873 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf867, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf872, (2048, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_27
        buf874 = buf873[0]
        assert_size_stride(buf874, (2048, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf874, 16, 'torch.ops.aten.convolution_backward.default')
        del buf873
        buf875 = buf872; del buf872  # reuse
        buf876 = reinterpret_tensor(buf826, (2048, 128, 28, 28), (100352, 784, 28, 1), 0); del buf826  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_53.run(buf870, convert_element_type_28, buf876, 205520896, stream=stream0)
        del convert_element_type_28
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf877 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf867, (2048, 512, 28, 28), (401408, 784, 28, 1), 0), buf876, reinterpret_tensor(buf875, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf878 = buf877[1]
        assert_size_stride(buf878, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf878, 16, 'torch.ops.aten.convolution_backward.default')
        del buf877
        buf879 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(buf878, buf879, 65536, stream=stream0)
        del buf878
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_167, buf5, 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del buf141
        del buf200
        del buf6
        del buf662
        del buf685
        del buf728
        del buf75
        del buf750
        del buf794
        del buf816
        del getitem_167
        buf881 = reinterpret_tensor(buf876, (401408, 512), (512, 1), 0); del buf876  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_162, getitem_163, getitem_164, buf881, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_162
        del getitem_163
        del getitem_164
        buf883 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf884 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf904 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_66.run(buf666, buf883, buf884, buf904, 128, stream=stream0)
        buf885 = reinterpret_tensor(buf870, (2048, 128, 784), (100352, 784, 1), 0); del buf870  # reuse
        buf889 = reinterpret_tensor(buf818, (2048, 128, 784), (100352, 784, 1), 0); del buf818  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_57.run(buf874, buf5, buf885, buf889, 262144, 784, stream=stream0)
        del buf5
        del buf874
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf881, (2048, 128, 784), (100352, 784, 1), 0), buf885, buf883, buf884, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        buf888 = buf885; del buf885  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf881, (2048, 128, 784), (100352, 784, 1), 0), buf889, rsqrt_12, primals_76, buf883, buf884, buf888, 1605632, 784, 100352, 784, 1024, 128, 1568, 1, stream=stream0)
        del buf881
        del buf889
        del primals_76
        del rsqrt_12
        buf891 = reinterpret_tensor(buf867, (1605632, 512), (512, 1), 0); del buf867  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_155, getitem_156, getitem_157, buf891, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_155
        del getitem_156
        del getitem_157
        buf893 = buf875; del buf875  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf894 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf888, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf893, (2048, 128, 56, 56), (0, 0, 0, 0), 0), convert_element_type_25, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_25
        buf895 = buf894[0]
        assert_size_stride(buf895, (2048, 128, 56, 56), (401408, 1, 7168, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf895, 16, 'torch.ops.aten.convolution_backward.default')
        del buf894
        buf896 = buf893; del buf893  # reuse
        buf897 = reinterpret_tensor(buf868, (2048, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf868  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_67.run(buf891, convert_element_type_26, buf897, 822083584, stream=stream0)
        del convert_element_type_26
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf898 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf888, (2048, 128, 28, 28), (100352, 784, 28, 1), 0), buf897, reinterpret_tensor(buf896, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        del buf888
        buf899 = buf898[1]
        assert_size_stride(buf899, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf899, 16, 'torch.ops.aten.convolution_backward.default')
        del buf898
        buf900 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_58.run(buf899, buf900, 147456, stream=stream0)
        del buf899
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_154, buf639, 16, 1, 512, 1, 1, 32, 16, 1605632, 1, 1, stream=stream0)
        del buf640
        del buf707
        del buf772
        del buf838
        del getitem_154
        buf902 = reinterpret_tensor(buf897, (1605632, 512), (512, 1), 0); del buf897  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_149, getitem_150, getitem_151, buf902, 32, 1, 512, 1, 2, 16, 32, 1605632, 1, 1, stream=stream0)
        del getitem_149
        del getitem_150
        del getitem_151
        buf905 = reinterpret_tensor(buf891, (2048, 128, 3136), (401408, 3136, 1), 0); del buf891  # reuse
        buf909 = reinterpret_tensor(buf861, (2048, 128, 3136), (401408, 3136, 1), 0); del buf861  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf895, buf639, buf905, buf909, 262144, 3136, stream=stream0)
        del buf639
        del buf895
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_18.run(reinterpret_tensor(buf902, (2048, 128, 3136), (401408, 3136, 1), 0), buf905, buf666, buf904, 6422528, 3136, 401408, 3136, 1024, 128, 6272, 1, stream=stream0)
        buf908 = buf905; del buf905  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_19.run(reinterpret_tensor(buf902, (2048, 128, 3136), (401408, 3136, 1), 0), buf909, rsqrt_11, primals_70, buf666, buf904, buf908, 6422528, 3136, 401408, 3136, 1024, 128, 6272, 1, stream=stream0)
        del buf902
        del buf909
        del primals_70
        del rsqrt_11
        buf913 = buf896; del buf896  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf914 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf908, (2048, 128, 56, 56), (401408, 3136, 56, 1), 0), reinterpret_tensor(buf913, (2048, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_23, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_23
        buf915 = buf914[0]
        assert_size_stride(buf915, (2048, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf915, 16, 'torch.ops.aten.convolution_backward.default')
        del buf914
        buf916 = buf913; del buf913  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf918 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf908, (2048, 128, 56, 56), (401408, 3136, 56, 1), 0), buf917, reinterpret_tensor(buf916, (128, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        del buf908
        buf919 = buf918[1]
        assert_size_stride(buf919, (128, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf919, 16, 'torch.ops.aten.convolution_backward.default')
        del buf918
        buf920 = empty_strided_cuda((128, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_69.run(buf919, buf920, 32768, stream=stream0)
        del buf919
        buf921 = empty_strided_cuda((3211264, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_380], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_70.run(buf921, 1644167168, stream=stream0)
        buf922 = empty_strided_cuda((1644167168, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_71.run(buf921, buf922, 1644167168, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_141, reinterpret_tensor(buf922, (3211264, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 3211264, 1, 1, stream=stream0)
        del getitem_141
        buf924 = reinterpret_tensor(buf917, (3211264, 512), (512, 1), 0); del buf917  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_136, getitem_137, getitem_138, buf924, 32, 1, 512, 1, 2, 16, 32, 3211264, 1, 1, stream=stream0)
        del getitem_136
        del getitem_137
        del getitem_138
        buf926 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf927 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_72.run(buf251, buf926, buf927, 256, stream=stream0)
        buf928 = reinterpret_tensor(buf857, (2048, 256, 3136), (802816, 3136, 1), 0); del buf857  # reuse
        buf932 = reinterpret_tensor(buf911, (2048, 256, 3136), (802816, 3136, 1), 0); del buf911  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_73.run(buf855, buf915, buf922, buf928, buf932, 524288, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf924, (2048, 256, 3136), (802816, 3136, 1), 0), buf928, buf926, buf927, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        buf931 = buf928; del buf928  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf924, (2048, 256, 3136), (802816, 3136, 1), 0), buf932, rsqrt_10, primals_64, buf926, buf927, buf931, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        del primals_64
        del rsqrt_10
        buf934 = reinterpret_tensor(buf626, (802816, 512), (512, 1), 0); del buf626  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_129, getitem_130, getitem_131, buf934, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_129
        del getitem_130
        del getitem_131
        buf936 = buf916; del buf916  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf937 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf931, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf936, (2048, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_21
        buf938 = buf937[0]
        assert_size_stride(buf938, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf938, 16, 'torch.ops.aten.convolution_backward.default')
        del buf937
        buf939 = buf936; del buf936  # reuse
        buf940 = reinterpret_tensor(buf627, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf627  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74.run(buf934, convert_element_type_22, buf940, 411041792, stream=stream0)
        del convert_element_type_22
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf941 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf931, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), buf940, reinterpret_tensor(buf939, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf942 = buf941[1]
        assert_size_stride(buf942, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf942, 16, 'torch.ops.aten.convolution_backward.default')
        del buf941
        buf943 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf942, buf943, 16384, stream=stream0)
        del buf942
        buf944 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        buf967 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf222, buf944, buf967, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_128, reinterpret_tensor(buf944, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_128
        buf946 = reinterpret_tensor(buf940, (802816, 512), (512, 1), 0); del buf940  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_123, getitem_124, getitem_125, buf946, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_123
        del getitem_124
        del getitem_125
        buf948 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_76.run(buf948, 64, stream=stream0)
        buf949 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf950 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf971 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf972 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_77.run(buf948, buf949, buf950, buf971, buf972, 64, stream=stream0)
        buf951 = reinterpret_tensor(buf934, (2048, 64, 3136), (200704, 3136, 1), 0); del buf934  # reuse
        buf955 = reinterpret_tensor(buf619, (2048, 64, 3136), (200704, 3136, 1), 0); del buf619  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf938, buf944, buf951, buf955, 131072, 3136, stream=stream0)
        del buf938
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf946, (2048, 64, 3136), (200704, 3136, 1), 0), buf951, buf949, buf950, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf954 = buf951; del buf951  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf946, (2048, 64, 3136), (200704, 3136, 1), 0), buf955, rsqrt_9, primals_58, buf949, buf950, buf954, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del primals_58
        del rsqrt_9
        buf957 = reinterpret_tensor(buf955, (802816, 512), (512, 1), 0); del buf955  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_116, getitem_117, getitem_118, buf957, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_116
        del getitem_117
        del getitem_118
        buf959 = buf939; del buf939  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf960 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf954, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf959, (2048, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_19, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_19
        buf961 = buf960[0]
        assert_size_stride(buf961, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf961, 16, 'torch.ops.aten.convolution_backward.default')
        del buf960
        buf962 = buf959; del buf959  # reuse
        buf963 = reinterpret_tensor(buf946, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf946  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74.run(buf957, convert_element_type_20, buf963, 411041792, stream=stream0)
        del convert_element_type_20
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf964 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf954, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf963, reinterpret_tensor(buf962, (64, 64, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf965 = buf964[1]
        assert_size_stride(buf965, (64, 64, 3, 3), (576, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf965, 16, 'torch.ops.aten.convolution_backward.default')
        del buf964
        buf966 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_79.run(buf965, buf966, 36864, stream=stream0)
        del buf965
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_115, reinterpret_tensor(buf967, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_115
        buf969 = reinterpret_tensor(buf963, (802816, 512), (512, 1), 0); del buf963  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_110, getitem_111, getitem_112, buf969, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_110
        del getitem_111
        del getitem_112
        buf973 = buf954; del buf954  # reuse
        buf977 = reinterpret_tensor(buf957, (2048, 64, 3136), (200704, 3136, 1), 0); del buf957  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf961, buf967, buf973, buf977, 131072, 3136, stream=stream0)
        del buf961
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf969, (2048, 64, 3136), (200704, 3136, 1), 0), buf973, buf971, buf972, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf976 = buf973; del buf973  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf969, (2048, 64, 3136), (200704, 3136, 1), 0), buf977, rsqrt_8, primals_52, buf971, buf972, buf976, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del primals_52
        del rsqrt_8
        buf979 = reinterpret_tensor(buf931, (3211264, 512), (512, 1), 0); del buf931  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_103, getitem_104, getitem_105, buf979, 32, 1, 512, 1, 2, 16, 32, 3211264, 1, 1, stream=stream0)
        del getitem_103
        del getitem_104
        del getitem_105
        buf981 = buf962; del buf962  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf982 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf976, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf981, (2048, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_17, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_17
        buf983 = buf982[0]
        assert_size_stride(buf983, (2048, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf983, 16, 'torch.ops.aten.convolution_backward.default')
        del buf982
        buf984 = buf981; del buf981  # reuse
        buf985 = reinterpret_tensor(buf932, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf932  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_80.run(buf979, convert_element_type_18, buf985, 1644167168, stream=stream0)
        del convert_element_type_18
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf986 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf976, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf985, reinterpret_tensor(buf984, (64, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf987 = buf986[1]
        assert_size_stride(buf987, (64, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf987, 16, 'torch.ops.aten.convolution_backward.default')
        del buf986
        buf988 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf987, buf988, 16384, stream=stream0)
        del buf987
        buf989 = empty_strided_cuda((1644167168, ), (1, ), torch.int8)
        buf1054 = empty_strided_cuda((1644167168, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_81.run(buf921, buf989, buf1054, 1644167168, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_102, reinterpret_tensor(buf989, (3211264, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 3211264, 1, 1, stream=stream0)
        del getitem_102
        buf991 = buf985; del buf985  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_82.run(buf855, buf915, buf922, buf983, buf989, buf991, 524288, 3136, stream=stream0)
        buf992 = reinterpret_tensor(buf983, (3211264, 512), (512, 1), 0); del buf983  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_97, getitem_98, getitem_99, buf992, 32, 1, 512, 1, 2, 16, 32, 3211264, 1, 1, stream=stream0)
        del getitem_97
        del getitem_98
        del getitem_99
        buf994 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf995 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_72.run(buf251, buf994, buf995, 256, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf992, (2048, 256, 3136), (802816, 3136, 1), 0), reinterpret_tensor(buf991, (2048, 256, 3136), (802816, 3136, 1), 0), buf994, buf995, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        buf998 = reinterpret_tensor(buf915, (2048, 256, 3136), (802816, 3136, 1), 0); del buf915  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf992, (2048, 256, 3136), (802816, 3136, 1), 0), reinterpret_tensor(buf991, (2048, 256, 3136), (802816, 3136, 1), 0), rsqrt_7, primals_46, buf994, buf995, buf998, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        del primals_46
        del rsqrt_7
        buf1000 = reinterpret_tensor(buf976, (802816, 512), (512, 1), 0); del buf976  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_90, getitem_91, getitem_92, buf1000, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_90
        del getitem_91
        del getitem_92
        buf1002 = buf984; del buf984  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1003 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf998, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1002, (2048, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_15, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_15
        buf1004 = buf1003[0]
        assert_size_stride(buf1004, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1004, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1003
        buf1005 = buf1002; del buf1002  # reuse
        buf1006 = reinterpret_tensor(buf977, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf977  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74.run(buf1000, convert_element_type_16, buf1006, 411041792, stream=stream0)
        del convert_element_type_16
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf1007 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf998, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), buf1006, reinterpret_tensor(buf1005, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1008 = buf1007[1]
        assert_size_stride(buf1008, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1008, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1007
        buf1009 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf1008, buf1009, 16384, stream=stream0)
        del buf1008
        buf1010 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        buf1032 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf222, buf1010, buf1032, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_89, reinterpret_tensor(buf1010, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_89
        buf1012 = reinterpret_tensor(buf1006, (802816, 512), (512, 1), 0); del buf1006  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_84, getitem_85, getitem_86, buf1012, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_84
        del getitem_85
        del getitem_86
        buf1014 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1015 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1036 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1037 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_77.run(buf948, buf1014, buf1015, buf1036, buf1037, 64, stream=stream0)
        buf1016 = reinterpret_tensor(buf1000, (2048, 64, 3136), (200704, 3136, 1), 0); del buf1000  # reuse
        buf1020 = reinterpret_tensor(buf969, (2048, 64, 3136), (200704, 3136, 1), 0); del buf969  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf1004, buf1010, buf1016, buf1020, 131072, 3136, stream=stream0)
        del buf1004
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf1012, (2048, 64, 3136), (200704, 3136, 1), 0), buf1016, buf1014, buf1015, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf1019 = buf1016; del buf1016  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf1012, (2048, 64, 3136), (200704, 3136, 1), 0), buf1020, rsqrt_6, primals_40, buf1014, buf1015, buf1019, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del primals_40
        del rsqrt_6
        buf1022 = reinterpret_tensor(buf1020, (802816, 512), (512, 1), 0); del buf1020  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_77, getitem_78, getitem_79, buf1022, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_77
        del getitem_78
        del getitem_79
        buf1024 = buf1005; del buf1005  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1025 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1019, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1024, (2048, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_13, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_13
        buf1026 = buf1025[0]
        assert_size_stride(buf1026, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1026, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1025
        buf1027 = buf1024; del buf1024  # reuse
        buf1028 = reinterpret_tensor(buf1012, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf1012  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74.run(buf1022, convert_element_type_14, buf1028, 411041792, stream=stream0)
        del convert_element_type_14
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf1029 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1019, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf1028, reinterpret_tensor(buf1027, (64, 64, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf1030 = buf1029[1]
        assert_size_stride(buf1030, (64, 64, 3, 3), (576, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1030, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1029
        buf1031 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_79.run(buf1030, buf1031, 36864, stream=stream0)
        del buf1030
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_76, reinterpret_tensor(buf1032, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_76
        buf1034 = reinterpret_tensor(buf1028, (802816, 512), (512, 1), 0); del buf1028  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_71, getitem_72, getitem_73, buf1034, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_71
        del getitem_72
        del getitem_73
        buf1038 = buf1019; del buf1019  # reuse
        buf1042 = reinterpret_tensor(buf1022, (2048, 64, 3136), (200704, 3136, 1), 0); del buf1022  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf1026, buf1032, buf1038, buf1042, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf1034, (2048, 64, 3136), (200704, 3136, 1), 0), buf1038, buf1036, buf1037, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf1041 = buf1038; del buf1038  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf1034, (2048, 64, 3136), (200704, 3136, 1), 0), buf1042, rsqrt_5, primals_34, buf1036, buf1037, buf1041, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del primals_34
        del rsqrt_5
        buf1044 = reinterpret_tensor(buf998, (3211264, 512), (512, 1), 0); del buf998  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_64, getitem_65, getitem_66, buf1044, 32, 1, 512, 1, 2, 16, 32, 3211264, 1, 1, stream=stream0)
        del getitem_64
        del getitem_65
        del getitem_66
        buf1046 = buf1027; del buf1027  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1047 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1041, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1046, (2048, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_11, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_11
        buf1048 = buf1047[0]
        assert_size_stride(buf1048, (2048, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1048, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1047
        buf1049 = buf1046; del buf1046  # reuse
        buf1050 = reinterpret_tensor(buf992, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf992  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_80.run(buf1044, convert_element_type_12, buf1050, 1644167168, stream=stream0)
        del convert_element_type_12
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf1051 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1041, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf1050, reinterpret_tensor(buf1049, (64, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1052 = buf1051[1]
        assert_size_stride(buf1052, (64, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1052, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1051
        buf1053 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf1052, buf1053, 16384, stream=stream0)
        del buf1052
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_63, reinterpret_tensor(buf1054, (3211264, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 3211264, 1, 1, stream=stream0)
        del getitem_63
        buf1056 = reinterpret_tensor(buf1050, (3211264, 512), (512, 1), 0); del buf1050  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_58, getitem_59, getitem_60, buf1056, 32, 1, 512, 1, 2, 16, 32, 3211264, 1, 1, stream=stream0)
        del getitem_58
        del getitem_59
        del getitem_60
        buf1058 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1059 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1078 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_83.run(buf251, buf1058, buf1059, buf1078, 256, stream=stream0)
        buf1060 = reinterpret_tensor(buf1044, (2048, 256, 3136), (802816, 3136, 1), 0); del buf1044  # reuse
        buf1064 = reinterpret_tensor(buf855, (2048, 256, 3136), (802816, 3136, 1), 0); del buf855  # reuse
        buf1079 = reinterpret_tensor(buf979, (2048, 256, 3136), (802816, 3136, 1), 0); del buf979  # reuse
        buf1083 = reinterpret_tensor(buf924, (2048, 256, 3136), (802816, 3136, 1), 0); del buf924  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_84.run(buf991, buf1048, buf1054, buf1060, buf1064, buf1079, buf1083, 524288, 3136, stream=stream0)
        del buf1048
        del buf991
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf1056, (2048, 256, 3136), (802816, 3136, 1), 0), buf1060, buf1058, buf1059, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        buf1063 = buf1060; del buf1060  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf1056, (2048, 256, 3136), (802816, 3136, 1), 0), buf1064, rsqrt_4, primals_28, buf1058, buf1059, buf1063, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        del buf1056
        del primals_28
        del rsqrt_4
        buf1066 = reinterpret_tensor(buf1041, (802816, 512), (512, 1), 0); del buf1041  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_51, getitem_52, getitem_53, buf1066, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_51
        del getitem_52
        del getitem_53
        buf1068 = buf1049; del buf1049  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1069 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1063, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1068, (2048, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_9
        buf1070 = buf1069[0]
        assert_size_stride(buf1070, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1070, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1069
        buf1071 = buf1068; del buf1068  # reuse
        buf1128 = reinterpret_tensor(buf1042, (802816, 512), (512, 1), 0); del buf1042  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_15, getitem_16, getitem_17, buf1128, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_15
        del getitem_16
        del getitem_17
        buf1072 = reinterpret_tensor(buf1034, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf1034  # reuse
        buf1134 = reinterpret_tensor(buf1026, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf1026  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_85.run(buf1066, convert_element_type_4, buf1128, buf1072, buf1134, 411041792, stream=stream0)
        del convert_element_type_4
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf1073 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1063, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), buf1072, reinterpret_tensor(buf1071, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1074 = buf1073[1]
        assert_size_stride(buf1074, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1074, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1073
        buf1075 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf1074, buf1075, 16384, stream=stream0)
        del buf1074
        buf1076 = reinterpret_tensor(buf1063, (3211264, 512), (512, 1), 0); del buf1063  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_48, getitem_49, getitem_50, buf1076, 32, 1, 512, 1, 2, 16, 32, 3211264, 1, 1, stream=stream0)
        del getitem_48
        del getitem_49
        del getitem_50
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf1076, (2048, 256, 3136), (802816, 3136, 1), 0), buf1079, buf251, buf1078, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        buf1082 = buf1079; del buf1079  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf1076, (2048, 256, 3136), (802816, 3136, 1), 0), buf1083, rsqrt_3, primals_22, buf251, buf1078, buf1082, 6422528, 3136, 802816, 3136, 1024, 256, 6272, 1, stream=stream0)
        del primals_22
        del rsqrt_3
        buf1085 = reinterpret_tensor(buf1072, (802816, 512), (512, 1), 0); del buf1072  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_41, getitem_42, getitem_43, buf1085, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_41
        del getitem_42
        del getitem_43
        buf1087 = buf1071; del buf1071  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1088 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1082, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1087, (2048, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_7, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_7
        buf1089 = buf1088[0]
        assert_size_stride(buf1089, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1089, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1088
        buf1090 = buf1087; del buf1087  # reuse
        buf1091 = reinterpret_tensor(buf1128, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf1128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74.run(buf1085, convert_element_type_8, buf1091, 411041792, stream=stream0)
        del convert_element_type_8
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf1092 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1082, (2048, 256, 56, 56), (802816, 3136, 56, 1), 0), buf1091, reinterpret_tensor(buf1090, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1093 = buf1092[1]
        assert_size_stride(buf1093, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1093, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1092
        buf1094 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf1093, buf1094, 16384, stream=stream0)
        del buf1093
        buf1095 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_25.run(buf222, buf1095, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_40, reinterpret_tensor(buf1095, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_40
        buf1097 = reinterpret_tensor(buf1091, (802816, 512), (512, 1), 0); del buf1091  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_35, getitem_36, getitem_37, buf1097, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_35
        del getitem_36
        del getitem_37
        buf1099 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1100 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1120 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1121 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1143 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_86.run(buf948, buf1099, buf1100, buf1120, buf1121, buf1143, 64, stream=stream0)
        buf1101 = reinterpret_tensor(buf1085, (2048, 64, 3136), (200704, 3136, 1), 0); del buf1085  # reuse
        buf1105 = reinterpret_tensor(buf1066, (2048, 64, 3136), (200704, 3136, 1), 0); del buf1066  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf1089, buf1095, buf1101, buf1105, 131072, 3136, stream=stream0)
        del buf1089
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf1097, (2048, 64, 3136), (200704, 3136, 1), 0), buf1101, buf1099, buf1100, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf1104 = buf1101; del buf1101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf1097, (2048, 64, 3136), (200704, 3136, 1), 0), buf1105, rsqrt_2, primals_16, buf1099, buf1100, buf1104, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del primals_16
        del rsqrt_2
        buf1107 = reinterpret_tensor(buf1105, (802816, 512), (512, 1), 0); del buf1105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_28, getitem_29, getitem_30, buf1107, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_28
        del getitem_29
        del getitem_30
        buf1109 = buf1090; del buf1090  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1110 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1104, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1109, (2048, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_5, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_5
        buf1111 = buf1110[0]
        assert_size_stride(buf1111, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1111, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1110
        buf1112 = buf1109; del buf1109  # reuse
        buf1113 = reinterpret_tensor(buf1097, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0); del buf1097  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_74.run(buf1107, convert_element_type_6, buf1113, 411041792, stream=stream0)
        del convert_element_type_6
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf1114 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1104, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf1113, reinterpret_tensor(buf1112, (64, 64, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf1115 = buf1114[1]
        assert_size_stride(buf1115, (64, 64, 3, 3), (576, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1115, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1114
        buf1116 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_79.run(buf1115, buf1116, 36864, stream=stream0)
        del buf1115
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_27, buf222, 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del buf1010
        del buf1032
        del buf1095
        del buf223
        del buf292
        del buf357
        del buf423
        del buf488
        del buf554
        del buf617
        del buf944
        del buf967
        del getitem_27
        buf1118 = reinterpret_tensor(buf1113, (802816, 512), (512, 1), 0); del buf1113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_22, getitem_23, getitem_24, buf1118, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_22
        del getitem_23
        del getitem_24
        buf1122 = buf1104; del buf1104  # reuse
        buf1126 = reinterpret_tensor(buf1107, (2048, 64, 3136), (200704, 3136, 1), 0); del buf1107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_78.run(buf1111, buf222, buf1122, buf1126, 131072, 3136, stream=stream0)
        del buf1111
        del buf222
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf1118, (2048, 64, 3136), (200704, 3136, 1), 0), buf1122, buf1120, buf1121, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        buf1125 = buf1122; del buf1122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf1118, (2048, 64, 3136), (200704, 3136, 1), 0), buf1126, rsqrt_1, primals_10, buf1120, buf1121, buf1125, 6422528, 3136, 200704, 3136, 1024, 64, 6272, 1, stream=stream0)
        del buf1118
        del buf1126
        del primals_10
        del rsqrt_1
        buf1130 = buf1112; del buf1112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1131 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1125, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1130, (2048, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_3, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_3
        buf1132 = buf1131[0]
        assert_size_stride(buf1132, (2048, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1132, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1131
        buf1133 = buf1130; del buf1130  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf1135 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1125, (2048, 64, 56, 56), (200704, 3136, 56, 1), 0), buf1134, reinterpret_tensor(buf1133, (64, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        del buf1125
        del buf1134
        buf1136 = buf1135[1]
        assert_size_stride(buf1136, (64, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1136, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1135
        buf1137 = empty_strided_cuda((64, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_87.run(buf1136, buf1137, 4096, stream=stream0)
        del buf1136
        buf1138 = buf1070; del buf1070  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_88.run(buf1138, buf1132, 411041792, stream=stream0)
        del buf1132
        buf1139 = reinterpret_tensor(buf1082, (2048, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf1082  # reuse
        # Topologically Sorted Source Nodes: [maxpool], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_89.run(getitem_14, buf1138, buf1139, 25690112, 64, stream=stream0)
        del buf1138
        del getitem_14
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_12, buf921, 16, 1, 512, 1, 1, 32, 16, 3211264, 1, 1, stream=stream0)
        del buf1054
        del buf922
        del buf989
        del getitem_12
        buf1141 = reinterpret_tensor(buf1083, (3211264, 512), (512, 1), 0); del buf1083  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_7, getitem_8, getitem_9, buf1141, 32, 1, 512, 1, 2, 16, 32, 3211264, 1, 1, stream=stream0)
        del getitem_7
        del getitem_8
        del getitem_9
        buf1144 = reinterpret_tensor(buf1076, (2048, 64, 12544), (802816, 12544, 1), 0); del buf1076  # reuse
        buf1148 = reinterpret_tensor(buf1064, (2048, 64, 12544), (802816, 12544, 1), 0); del buf1064  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_90.run(buf1139, buf921, buf1144, buf1148, 1644167168, stream=stream0)
        del buf1139
        del buf921
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_24.run(reinterpret_tensor(buf1141, (2048, 64, 12544), (802816, 12544, 1), 0), buf1144, buf948, buf1143, 25690112, 12544, 802816, 12544, 1024, 64, 25088, 1, stream=stream0)
        buf1147 = buf1144; del buf1144  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_25.run(reinterpret_tensor(buf1141, (2048, 64, 12544), (802816, 12544, 1), 0), buf1148, rsqrt, primals_4, buf948, buf1143, buf1147, 25690112, 12544, 802816, 12544, 1024, 64, 25088, 1, stream=stream0)
        del buf1141
        del buf1148
        del primals_4
        del rsqrt
        buf1150 = empty_strided_cuda((602112, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem, getitem_1, getitem_2, buf1150, 32, 1, 512, 1, 2, 16, 32, 602112, 1, 1, stream=stream0)
        del getitem
        del getitem_1
        del getitem_2
        buf1152 = buf1133; del buf1133  # reuse
        buf1153 = empty_strided_cuda((2048, 3, 224, 224), (150528, 50176, 224, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_convolution_backward_91.run(buf1150, convert_element_type_2, buf1153, 308281344, stream=stream0)
        del buf1150
        del convert_element_type_2
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._unsafe_index, aten._to_copy, aten.add, aten.convolution_backward]
        buf1154 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1147, (2048, 64, 112, 112), (802816, 12544, 112, 1), 0), buf1153, reinterpret_tensor(buf1152, (64, 3, 7, 7), (0, 0, 0, 0), 0), None, [2, 2], [3, 3], [1, 1], False, [0], 1, [False, True, False])
        del buf1147
        del buf1152
        del buf1153
        buf1155 = buf1154[1]
        assert_size_stride(buf1155, (64, 3, 7, 7), (147, 49, 7, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1155, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1154
        buf1156 = empty_strided_cuda((64, 3, 7, 7), (147, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_92.run(buf1155, buf1156, 9408, stream=stream0)
        del buf1155
    return (buf1156, None, None, buf1143, buf948, None, None, buf1137, None, buf1121, buf1120, None, None, buf1116, None, buf1100, buf1099, None, None, buf1094, None, buf1078, buf251, None, None, buf1075, None, buf1059, buf1058, None, None, buf1053, None, buf1037, buf1036, None, None, buf1031, None, buf1015, buf1014, None, None, buf1009, None, buf995, buf994, None, None, buf988, None, buf972, buf971, None, None, buf966, None, buf950, buf949, None, None, buf943, None, buf927, buf926, None, None, buf920, None, buf904, buf666, None, None, buf900, None, buf884, buf883, None, None, buf879, None, buf863, buf34, None, None, buf860, None, buf844, buf843, None, None, buf837, None, buf821, buf820, None, None, buf815, None, buf799, buf798, None, None, buf793, None, buf777, buf776, None, None, buf771, None, buf755, buf754, None, None, buf749, None, buf733, buf732, None, None, buf727, None, buf713, buf712, None, None, buf706, None, buf690, buf689, None, None, buf684, None, buf668, buf667, None, None, buf661, None, buf645, buf644, None, None, buf638, None, buf622, buf621, None, None, buf616, None, buf600, buf599, None, None, buf595, None, buf579, buf227, None, None, buf576, None, buf560, buf559, None, None, buf553, None, buf537, buf536, None, None, buf531, None, buf515, buf514, None, None, buf509, None, buf493, buf492, None, None, buf487, None, buf471, buf470, None, None, buf465, None, buf449, buf448, None, None, buf443, None, buf429, buf428, None, None, buf422, None, buf406, buf405, None, None, buf400, None, buf384, buf383, None, None, buf378, None, buf362, buf361, None, None, buf356, None, buf340, buf339, None, None, buf334, None, buf318, buf317, None, None, buf312, None, buf298, buf297, None, None, buf291, None, buf275, buf274, None, None, buf269, None, buf253, buf252, None, None, buf245, None, buf229, buf228, None, None, buf221, None, buf205, buf204, None, None, buf199, None, buf183, buf182, None, None, buf178, None, buf164, buf10, None, None, buf161, None, buf147, buf146, None, None, buf140, None, buf124, buf123, None, None, buf118, None, buf102, buf101, None, None, buf96, None, buf80, buf79, None, None, buf74, None, buf58, buf57, None, None, buf52, None, buf36, buf35, None, None, buf28, None, buf12, buf11, None, None, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_2 = rand_strided((2048, 3, 74, 74), (16428, 1, 222, 3), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem = rand_strided((602112, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_1 = rand_strided((602112, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_2 = rand_strided((602112, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((3211264, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_8 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_10 = rand_strided((2048, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_12 = rand_strided((3211264, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_14 = rand_strided((2048, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.int8)
    convert_element_type_3 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_4 = rand_strided((2048, 64, 18, 18), (20736, 1, 1152, 64), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_15 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_16 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_17 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_22 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_23 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_24 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_27 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_5 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_6 = rand_strided((2048, 64, 18, 18), (20736, 1, 1152, 64), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_28 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_29 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_30 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_36 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_37 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_40 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_7 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_8 = rand_strided((2048, 64, 18, 18), (20736, 1, 1152, 64), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_41 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_42 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_43 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_48 = rand_strided((3211264, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_49 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_50 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_9 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_51 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_52 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_53 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((3211264, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_59 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_60 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_63 = rand_strided((3211264, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_11 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_12 = rand_strided((2048, 256, 18, 18), (82944, 1, 4608, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_64 = rand_strided((3211264, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_65 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_66 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_72 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_73 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_76 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_13 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_14 = rand_strided((2048, 64, 18, 18), (20736, 1, 1152, 64), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_77 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_78 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_79 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_85 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_86 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_89 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_15 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_16 = rand_strided((2048, 64, 18, 18), (20736, 1, 1152, 64), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_90 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_91 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_92 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_7 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((3211264, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_98 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_99 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_102 = rand_strided((3211264, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_17 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_18 = rand_strided((2048, 256, 18, 18), (82944, 1, 4608, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_103 = rand_strided((3211264, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_104 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_105 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_110 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_111 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_112 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_115 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_19 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_20 = rand_strided((2048, 64, 18, 18), (20736, 1, 1152, 64), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_116 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_117 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_118 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_124 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_125 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_128 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_21 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_22 = rand_strided((2048, 64, 18, 18), (20736, 1, 1152, 64), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_129 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_130 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_131 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_136 = rand_strided((3211264, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_137 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_138 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_141 = rand_strided((3211264, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_23 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_24 = rand_strided((2048, 256, 18, 18), (82944, 1, 4608, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_142 = rand_strided((3211264, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_143 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_144 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_150 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_151 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_154 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_25 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_26 = rand_strided((2048, 128, 18, 18), (41472, 1, 2304, 128), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_155 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_156 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_157 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_162 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_163 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_164 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_167 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_27 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_28 = rand_strided((2048, 128, 9, 9), (10368, 1, 1152, 128), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_168 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_169 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_170 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_13 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_175 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_176 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_177 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_29 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_178 = rand_strided((3211264, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_179 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_180 = rand_strided((3211264, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_14 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_185 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_186 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_187 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_190 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_31 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_32 = rand_strided((2048, 512, 9, 9), (41472, 1, 4608, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_191 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_192 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_193 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_198 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_199 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_200 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_203 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_33 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_34 = rand_strided((2048, 128, 9, 9), (10368, 1, 1152, 128), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_204 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_205 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_206 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_211 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_212 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_213 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_216 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_35 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_36 = rand_strided((2048, 128, 9, 9), (10368, 1, 1152, 128), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_217 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_218 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_219 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_224 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_225 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_226 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_229 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_37 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_38 = rand_strided((2048, 512, 9, 9), (41472, 1, 4608, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_230 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_231 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_232 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_237 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_238 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_239 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_242 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_39 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_40 = rand_strided((2048, 128, 9, 9), (10368, 1, 1152, 128), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_243 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_244 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_245 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_250 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_251 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_252 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_255 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_41 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_42 = rand_strided((2048, 128, 9, 9), (10368, 1, 1152, 128), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_256 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_257 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_258 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_263 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_264 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_265 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_268 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_43 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_44 = rand_strided((2048, 512, 9, 9), (41472, 1, 4608, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_269 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_270 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_271 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_276 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_277 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_278 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_281 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_45 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_46 = rand_strided((2048, 128, 9, 9), (10368, 1, 1152, 128), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_282 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_283 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_284 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_289 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_290 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_291 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_294 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_47 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_48 = rand_strided((2048, 128, 9, 9), (10368, 1, 1152, 128), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_295 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_296 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_297 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_302 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_303 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_304 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_307 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_49 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_50 = rand_strided((2048, 512, 9, 9), (41472, 1, 4608, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_308 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_309 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_310 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_315 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_316 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_317 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_320 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_51 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_52 = rand_strided((2048, 256, 9, 9), (20736, 1, 2304, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_321 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_322 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_323 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_328 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_329 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_330 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_333 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_53 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_54 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_334 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_335 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_336 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_26 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_341 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_342 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_343 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_55 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_344 = rand_strided((1605632, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_345 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_346 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_27 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_351 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_352 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_353 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_356 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_57 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_58 = rand_strided((2048, 1024, 4, 4), (16384, 1, 4096, 1024), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_357 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_358 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_359 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_364 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_365 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_366 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_369 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_59 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_60 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_370 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_371 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_372 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_377 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_378 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_379 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_382 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_61 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_62 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_383 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_384 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_385 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_30 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_390 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_391 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_392 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_395 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_63 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_64 = rand_strided((2048, 1024, 4, 4), (16384, 1, 4096, 1024), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_396 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_397 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_398 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_403 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_404 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_405 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_408 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_65 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_66 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_409 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_410 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_411 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_416 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_417 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_418 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_421 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_67 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_68 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_422 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_423 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_424 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_33 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_429 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_430 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_431 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_434 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_69 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_70 = rand_strided((2048, 1024, 4, 4), (16384, 1, 4096, 1024), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_435 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_436 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_437 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_442 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_443 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_444 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_447 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_71 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_72 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_448 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_449 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_450 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_455 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_456 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_457 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_460 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_73 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_74 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_461 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_462 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_463 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_36 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_468 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_469 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_470 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_473 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_75 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_76 = rand_strided((2048, 1024, 4, 4), (16384, 1, 4096, 1024), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_474 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_475 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_476 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_481 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_482 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_483 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_486 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_77 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_78 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_487 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_488 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_489 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_494 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_495 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_496 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_499 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_79 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_80 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_500 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_501 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_502 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_39 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_507 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_508 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_509 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_512 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_81 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_82 = rand_strided((2048, 1024, 4, 4), (16384, 1, 4096, 1024), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_513 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_514 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_515 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_520 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_521 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_522 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_525 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_83 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_84 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_526 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_527 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_528 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_533 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_534 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_535 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_538 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_85 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_86 = rand_strided((2048, 256, 4, 4), (4096, 1, 1024, 256), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_539 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_540 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_541 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_42 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_546 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_547 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_548 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_551 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_87 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_88 = rand_strided((2048, 1024, 4, 4), (16384, 1, 4096, 1024), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_552 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_553 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_554 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_559 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_560 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_561 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_564 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_89 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_90 = rand_strided((2048, 512, 4, 4), (8192, 1, 2048, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_565 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_566 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_567 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_572 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_573 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_574 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_577 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_91 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_92 = rand_strided((2048, 512, 2, 2), (2048, 1, 1024, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_578 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_579 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_580 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_45 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_585 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_586 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_587 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_93 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_588 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_589 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_590 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_46 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_595 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_596 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_597 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_600 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_95 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_96 = rand_strided((2048, 2048, 2, 2), (8192, 1, 4096, 2048), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_601 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_602 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_603 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_47 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_608 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_609 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_610 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_613 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_97 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_98 = rand_strided((2048, 512, 2, 2), (2048, 1, 1024, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_614 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_615 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_616 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_621 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_622 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_623 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_626 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_99 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_100 = rand_strided((2048, 512, 2, 2), (2048, 1, 1024, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_627 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_628 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_629 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_49 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_634 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_635 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_636 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_639 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_101 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_102 = rand_strided((2048, 2048, 2, 2), (8192, 1, 4096, 2048), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_640 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_641 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_642 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_647 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_648 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_649 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_652 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_103 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_104 = rand_strided((2048, 512, 2, 2), (2048, 1, 1024, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_653 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_654 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_655 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_660 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_661 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_662 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_665 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_105 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_106 = rand_strided((2048, 512, 2, 2), (2048, 1, 1024, 512), device='cuda:0', dtype=torch.float8_e4m3fn)
    getitem_666 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_667 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_668 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_52 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_673 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_674 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_675 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_678 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_679 = rand_strided((8192, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_680 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_681 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_107 = rand_strided((100, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((2048, 100), (100, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, convert_element_type_2, getitem, getitem_1, getitem_2, rsqrt, getitem_7, getitem_8, getitem_9, getitem_10, getitem_12, getitem_14, convert_element_type_3, convert_element_type_4, getitem_15, getitem_16, getitem_17, rsqrt_1, getitem_22, getitem_23, getitem_24, getitem_27, convert_element_type_5, convert_element_type_6, getitem_28, getitem_29, getitem_30, rsqrt_2, getitem_35, getitem_36, getitem_37, getitem_40, convert_element_type_7, convert_element_type_8, getitem_41, getitem_42, getitem_43, rsqrt_3, getitem_48, getitem_49, getitem_50, convert_element_type_9, getitem_51, getitem_52, getitem_53, rsqrt_4, getitem_58, getitem_59, getitem_60, getitem_63, convert_element_type_11, convert_element_type_12, getitem_64, getitem_65, getitem_66, rsqrt_5, getitem_71, getitem_72, getitem_73, getitem_76, convert_element_type_13, convert_element_type_14, getitem_77, getitem_78, getitem_79, rsqrt_6, getitem_84, getitem_85, getitem_86, getitem_89, convert_element_type_15, convert_element_type_16, getitem_90, getitem_91, getitem_92, rsqrt_7, getitem_97, getitem_98, getitem_99, getitem_102, convert_element_type_17, convert_element_type_18, getitem_103, getitem_104, getitem_105, rsqrt_8, getitem_110, getitem_111, getitem_112, getitem_115, convert_element_type_19, convert_element_type_20, getitem_116, getitem_117, getitem_118, rsqrt_9, getitem_123, getitem_124, getitem_125, getitem_128, convert_element_type_21, convert_element_type_22, getitem_129, getitem_130, getitem_131, rsqrt_10, getitem_136, getitem_137, getitem_138, getitem_141, convert_element_type_23, convert_element_type_24, getitem_142, getitem_143, getitem_144, rsqrt_11, getitem_149, getitem_150, getitem_151, getitem_154, convert_element_type_25, convert_element_type_26, getitem_155, getitem_156, getitem_157, rsqrt_12, getitem_162, getitem_163, getitem_164, getitem_167, convert_element_type_27, convert_element_type_28, getitem_168, getitem_169, getitem_170, rsqrt_13, getitem_175, getitem_176, getitem_177, convert_element_type_29, getitem_178, getitem_179, getitem_180, rsqrt_14, getitem_185, getitem_186, getitem_187, getitem_190, convert_element_type_31, convert_element_type_32, getitem_191, getitem_192, getitem_193, rsqrt_15, getitem_198, getitem_199, getitem_200, getitem_203, convert_element_type_33, convert_element_type_34, getitem_204, getitem_205, getitem_206, rsqrt_16, getitem_211, getitem_212, getitem_213, getitem_216, convert_element_type_35, convert_element_type_36, getitem_217, getitem_218, getitem_219, rsqrt_17, getitem_224, getitem_225, getitem_226, getitem_229, convert_element_type_37, convert_element_type_38, getitem_230, getitem_231, getitem_232, rsqrt_18, getitem_237, getitem_238, getitem_239, getitem_242, convert_element_type_39, convert_element_type_40, getitem_243, getitem_244, getitem_245, rsqrt_19, getitem_250, getitem_251, getitem_252, getitem_255, convert_element_type_41, convert_element_type_42, getitem_256, getitem_257, getitem_258, rsqrt_20, getitem_263, getitem_264, getitem_265, getitem_268, convert_element_type_43, convert_element_type_44, getitem_269, getitem_270, getitem_271, rsqrt_21, getitem_276, getitem_277, getitem_278, getitem_281, convert_element_type_45, convert_element_type_46, getitem_282, getitem_283, getitem_284, rsqrt_22, getitem_289, getitem_290, getitem_291, getitem_294, convert_element_type_47, convert_element_type_48, getitem_295, getitem_296, getitem_297, rsqrt_23, getitem_302, getitem_303, getitem_304, getitem_307, convert_element_type_49, convert_element_type_50, getitem_308, getitem_309, getitem_310, rsqrt_24, getitem_315, getitem_316, getitem_317, getitem_320, convert_element_type_51, convert_element_type_52, getitem_321, getitem_322, getitem_323, rsqrt_25, getitem_328, getitem_329, getitem_330, getitem_333, convert_element_type_53, convert_element_type_54, getitem_334, getitem_335, getitem_336, rsqrt_26, getitem_341, getitem_342, getitem_343, convert_element_type_55, getitem_344, getitem_345, getitem_346, rsqrt_27, getitem_351, getitem_352, getitem_353, getitem_356, convert_element_type_57, convert_element_type_58, getitem_357, getitem_358, getitem_359, rsqrt_28, getitem_364, getitem_365, getitem_366, getitem_369, convert_element_type_59, convert_element_type_60, getitem_370, getitem_371, getitem_372, rsqrt_29, getitem_377, getitem_378, getitem_379, getitem_382, convert_element_type_61, convert_element_type_62, getitem_383, getitem_384, getitem_385, rsqrt_30, getitem_390, getitem_391, getitem_392, getitem_395, convert_element_type_63, convert_element_type_64, getitem_396, getitem_397, getitem_398, rsqrt_31, getitem_403, getitem_404, getitem_405, getitem_408, convert_element_type_65, convert_element_type_66, getitem_409, getitem_410, getitem_411, rsqrt_32, getitem_416, getitem_417, getitem_418, getitem_421, convert_element_type_67, convert_element_type_68, getitem_422, getitem_423, getitem_424, rsqrt_33, getitem_429, getitem_430, getitem_431, getitem_434, convert_element_type_69, convert_element_type_70, getitem_435, getitem_436, getitem_437, rsqrt_34, getitem_442, getitem_443, getitem_444, getitem_447, convert_element_type_71, convert_element_type_72, getitem_448, getitem_449, getitem_450, rsqrt_35, getitem_455, getitem_456, getitem_457, getitem_460, convert_element_type_73, convert_element_type_74, getitem_461, getitem_462, getitem_463, rsqrt_36, getitem_468, getitem_469, getitem_470, getitem_473, convert_element_type_75, convert_element_type_76, getitem_474, getitem_475, getitem_476, rsqrt_37, getitem_481, getitem_482, getitem_483, getitem_486, convert_element_type_77, convert_element_type_78, getitem_487, getitem_488, getitem_489, rsqrt_38, getitem_494, getitem_495, getitem_496, getitem_499, convert_element_type_79, convert_element_type_80, getitem_500, getitem_501, getitem_502, rsqrt_39, getitem_507, getitem_508, getitem_509, getitem_512, convert_element_type_81, convert_element_type_82, getitem_513, getitem_514, getitem_515, rsqrt_40, getitem_520, getitem_521, getitem_522, getitem_525, convert_element_type_83, convert_element_type_84, getitem_526, getitem_527, getitem_528, rsqrt_41, getitem_533, getitem_534, getitem_535, getitem_538, convert_element_type_85, convert_element_type_86, getitem_539, getitem_540, getitem_541, rsqrt_42, getitem_546, getitem_547, getitem_548, getitem_551, convert_element_type_87, convert_element_type_88, getitem_552, getitem_553, getitem_554, rsqrt_43, getitem_559, getitem_560, getitem_561, getitem_564, convert_element_type_89, convert_element_type_90, getitem_565, getitem_566, getitem_567, rsqrt_44, getitem_572, getitem_573, getitem_574, getitem_577, convert_element_type_91, convert_element_type_92, getitem_578, getitem_579, getitem_580, rsqrt_45, getitem_585, getitem_586, getitem_587, convert_element_type_93, getitem_588, getitem_589, getitem_590, rsqrt_46, getitem_595, getitem_596, getitem_597, getitem_600, convert_element_type_95, convert_element_type_96, getitem_601, getitem_602, getitem_603, rsqrt_47, getitem_608, getitem_609, getitem_610, getitem_613, convert_element_type_97, convert_element_type_98, getitem_614, getitem_615, getitem_616, rsqrt_48, getitem_621, getitem_622, getitem_623, getitem_626, convert_element_type_99, convert_element_type_100, getitem_627, getitem_628, getitem_629, rsqrt_49, getitem_634, getitem_635, getitem_636, getitem_639, convert_element_type_101, convert_element_type_102, getitem_640, getitem_641, getitem_642, rsqrt_50, getitem_647, getitem_648, getitem_649, getitem_652, convert_element_type_103, convert_element_type_104, getitem_653, getitem_654, getitem_655, rsqrt_51, getitem_660, getitem_661, getitem_662, getitem_665, convert_element_type_105, convert_element_type_106, getitem_666, getitem_667, getitem_668, rsqrt_52, getitem_673, getitem_674, getitem_675, getitem_678, getitem_679, getitem_680, getitem_681, convert_element_type_107, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
