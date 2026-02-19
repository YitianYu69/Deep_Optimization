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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/act_triton_kernel.py:87
dequant_unpack_kernel_0 = async_compile.triton('dequant_unpack_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'dequant_unpack_kernel_0', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'P_ptr': '*i32', 'S_ptr': '*bf16', 'M_ptr': '*bf16', 'Y_ptr': '*bf16', 'stride_p0': 'constexpr', 'stride_p1': 'constexpr', 'stride_y0': 'constexpr', 'stride_y1': 'constexpr', 'BITS': 'constexpr', 'VPW': 'constexpr', 'NWORDS': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 16}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
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
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_1, torch.float32), kwargs = {})
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


# kernel path: /tmp/torchinductor_yyu496/fh/cfhqxn6xy5vyxuwu4o5rlxsoeyweqe2a3ssek5lcpwhbwyfcxeth.py
# Topologically Sorted Source Nodes: [full_257], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_257 => full_default_310
# Graph fragment:
#   %full_default_310 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([200704, 256], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_1 = async_compile.triton('triton_poi_fused_zeros_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/uf/cufyf44eedngq534urrlzon376smmtdz2lo2bcrju7dz7wukngfk.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_133, clone_default_142
# Graph fragment:
#   %clone_default_142 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_284,), kwargs = {})
#   %clone_default_133 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_266,), kwargs = {})
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256901120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/act_triton_kernel.py:183
unpack_kernel_1 = async_compile.triton('unpack_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'unpack_kernel_1', 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'P_ptr': '*i32', 'Y_ptr': '*i8', 'stride_p0': 'constexpr', 'stride_p1': 'constexpr', 'stride_y0': 'constexpr', 'stride_y1': 'constexpr', 'BITS': 'constexpr', 'VPW': 'constexpr', 'NWORDS': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'stride_p0': 8, 'stride_p1': 1, 'stride_y0': 256, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 8}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
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


# kernel path: /tmp/torchinductor_yyu496/mk/cmkz3jpivp3mu5oinb3y37hmit7o53ptcnlpu5ywhom3dfc2264f.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_131, clone_default_132, clone_default_140, clone_default_141
# Graph fragment:
#   %clone_default_140 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_280,), kwargs = {})
#   %clone_default_141 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_282,), kwargs = {})
#   %clone_default_131 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_262,), kwargs = {})
#   %clone_default_132 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_264,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 73728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/bk/cbk2dx6w6765i3fzqhwbeqgiosge3prie33fc44ml66mwlx7efpv.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_257, triton_kernel_wrapper_mutation_258
# Graph fragment:
#   %triton_kernel_wrapper_mutation_258 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 436, constant_args_idx: 625, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1669, DY: %view_1666, DBETA: %as_strided_default_281, DGAMMA: %as_strided_default_283, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_257 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 438, constant_args_idx: 626, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1669, DY: %view_1666, INVSTD: %rsqrt_52, GAMMA: %primals_316, DBETA: %as_strided_default_281, DGAMMA: %as_strided_default_283, DX: %permute_157, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/c5/cc5eyokpb6ygsxksmly7cveyvkokxdgsmmcbws2ilqumro5tx4fp.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_67 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_693, torch.float32), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fs/cfsb2635ptzwybzuovkr5q7gu55m3cdp6eqk33nw3w7bxczrkmdq.py
# Topologically Sorted Source Nodes: [full_260], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_260 => full_default_313
# Graph fragment:
#   %full_default_313 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([50176, 256], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_7 = async_compile.triton('triton_poi_fused_zeros_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25690112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fm/cfmv7uumddocmuvcnbop2lsewblwmuh47ojdtm6l76wjqdeskv5z.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_136, clone_default_139
# Graph fragment:
#   %clone_default_139 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_278,), kwargs = {})
#   %clone_default_136 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_272,), kwargs = {})
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64225280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/cc/ccctfxyj27vmwsdianm3s4el3ipawsua7vmowojvxvytpq5mpkxj.py
# Topologically Sorted Source Nodes: [layer2_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_bn3 => full_default_76
# Graph fragment:
#   %full_default_76 : [num_users=24] = call_function[target=torch.ops.aten.full.default](args = ([512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_9 = async_compile.triton('triton_poi_fused_zeros_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yg/cygvvwwu2mfvq3ravdonzm2kvsi2kjvbiur6mxf53bv5bweicnms.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_134, clone_default_135, clone_default_137, clone_default_138
# Graph fragment:
#   %clone_default_137 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_274,), kwargs = {})
#   %clone_default_138 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_276,), kwargs = {})
#   %clone_default_134 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_268,), kwargs = {})
#   %clone_default_135 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_270,), kwargs = {})
triton_poi_fused_10 = async_compile.triton('triton_poi_fused_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_10(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/my/cmy25dqdqyime55et6dsvmszrfg7kzrcypxvks4lur36s5ur22pq.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_252, triton_kernel_wrapper_mutation_253
# Graph fragment:
#   %triton_kernel_wrapper_mutation_253 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 441, constant_args_idx: 630, grid: [(512, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1714, DY: %view_1711, DBETA: %as_strided_default_275, DGAMMA: %as_strided_default_277, M: 25088, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_252 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 442, constant_args_idx: 631, grid: [(512, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1714, DY: %view_1711, INVSTD: %rsqrt_51, GAMMA: %primals_310, DBETA: %as_strided_default_275, DGAMMA: %as_strided_default_277, DX: %permute_158, M: 25088, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_11 = async_compile.triton('triton_poi_fused_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 25690112, 'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_11(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 49*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 49*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/2a/c2akaq25pmxxkzfe6tqbe2pw3cm7dx2ek7kb5c6y7uox4xink7r2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_72 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_705, torch.float32), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/dh/cdhe2a6iiqxl2yea5sqa4zipk3j2lmkc72ob26za4ycbvt2emi4u.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_242, triton_kernel_wrapper_mutation_243
# Graph fragment:
#   %triton_kernel_wrapper_mutation_243 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 449, constant_args_idx: 640, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1804, DY: %view_1801, DBETA: %as_strided_default_263, DGAMMA: %as_strided_default_265, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_242 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 450, constant_args_idx: 641, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1804, DY: %view_1801, INVSTD: %rsqrt_49, GAMMA: %primals_298, DBETA: %as_strided_default_263, DGAMMA: %as_strided_default_265, DX: %permute_160, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_13 = async_compile.triton('triton_poi_fused_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 104857600, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
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


# kernel path: /tmp/torchinductor_yyu496/fi/cfiav3rfhr5rqtin5vzgfj3zf7lnnilf6ilkrgc7evomk7hdofld.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_159 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %mul_371 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_159, %view_1650), kwargs = {})
#   %add_228 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %getitem_713), kwargs = {})
#   %mul_374 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_228, %view_1785), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_374, %getitem_749), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_229, %view_1920), kwargs = {})
triton_poi_fused_add_div_mul_14 = async_compile.triton('triton_poi_fused_add_div_mul_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'in_ptr4': '*bf16', 'in_ptr5': '*i8', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 207618048, 'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
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


# kernel path: /tmp/torchinductor_yyu496/mm/cmmbdwuu3xvijxamch3lkpigyw25ea3ynmf7aiir2g3ve3oznbe3.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_121, clone_default_122, clone_default_123
# Graph fragment:
#   %clone_default_122 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_244,), kwargs = {})
#   %clone_default_123 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_246,), kwargs = {})
#   %clone_default_121 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_242,), kwargs = {})
triton_poi_fused_15 = async_compile.triton('triton_poi_fused_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 57344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_15(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/kn/cknq36xbzmd3ajl3mfk35q7tmva6khklhbrkmmffsbhz23zm5pxa.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_97 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_765, torch.float32), kwargs = {})
triton_poi_fused__to_copy_16 = async_compile.triton('triton_poi_fused__to_copy_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20971520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/52/c52ohpkn3lgsq7me3kzqt7hsxgvr7aphtfal5vjhkeb3snmsqdum.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_213, triton_kernel_wrapper_mutation_214
# Graph fragment:
#   %triton_kernel_wrapper_mutation_214 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 473, constant_args_idx: 669, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2068, DY: %view_2065, DBETA: %as_strided_default_233, DGAMMA: %as_strided_default_235, M: 100352, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_213 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 474, constant_args_idx: 670, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2068, DY: %view_2065, INVSTD: %rsqrt_43, GAMMA: %primals_262, DBETA: %as_strided_default_233, DGAMMA: %as_strided_default_235, DX: %permute_166, M: 100352, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_17 = async_compile.triton('triton_poi_fused_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 102760448, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_17(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 196*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/ho/chomcwnqaqnimckdqcmypgxn6yyb5gua7uuqusfjdcj7gbwfm3h3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_112 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_800, torch.float32), kwargs = {})
triton_poi_fused__to_copy_18 = async_compile.triton('triton_poi_fused__to_copy_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5242880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fr/cfrr5jw3zlpjkmyi5r6hlf74mwsd5ttyppml5qthbfuue4k36pys.py
# Topologically Sorted Source Nodes: [full_286], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_286 => full_default_339
# Graph fragment:
#   %full_default_339 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([401408, 256], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_19 = async_compile.triton('triton_poi_fused_zeros_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_19(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/6e/c6exzri6edafd3hq3myfl3w6eiol2fkxesyyohqyr7etslbhamyw.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_115
# Graph fragment:
#   %clone_default_115 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_230,), kwargs = {})
triton_poi_fused_20 = async_compile.triton('triton_poi_fused_20', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 308281344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/iz/ciz754vso3ntehbhzkqrx3prdv4aexws3g2ad62gg2fwpwebmsvy.py
# Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer3_0_bn3 => full_default_152
# Graph fragment:
#   %full_default_152 : [num_users=16] = call_function[target=torch.ops.aten.full.default](args = ([1024], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_21 = async_compile.triton('triton_poi_fused_zeros_21', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_21(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/b5/cb5kps7f6rq35vsjyo2rkuawdks3hue62366wfaf2qrwpwwlwdbm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_113, clone_default_114
# Graph fragment:
#   %clone_default_113 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_226,), kwargs = {})
#   %clone_default_114 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_228,), kwargs = {})
triton_poi_fused_22 = async_compile.triton('triton_poi_fused_22', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_22(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ij/cijkf3j22g7agvwetnhyeyt2e76owf3bpbxkhlh3qfmlud7qwlba.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_208, triton_kernel_wrapper_mutation_209
# Graph fragment:
#   %triton_kernel_wrapper_mutation_209 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 477, constant_args_idx: 674, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2113, DY: %view_2110, DBETA: %as_strided_default_227, DGAMMA: %as_strided_default_229, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_208 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 478, constant_args_idx: 675, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2113, DY: %view_2110, INVSTD: %rsqrt_42, GAMMA: %primals_256, DBETA: %as_strided_default_227, DGAMMA: %as_strided_default_229, DX: %permute_167, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_23 = async_compile.triton('triton_poi_fused_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 196*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/de/cdeyfcmnizplaiziwl5sfuxavsyczxvfgopgq5ytz6bpl6vefc2a.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_117 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_812, torch.float32), kwargs = {})
triton_poi_fused__to_copy_24 = async_compile.triton('triton_poi_fused__to_copy_24', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2621440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/h3/ch3zgwcsod27jhpyecnp6stdgtcszvo65kxazedf6optlxo4sbma.py
# Topologically Sorted Source Nodes: [full_289], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_289 => full_default_342
# Graph fragment:
#   %full_default_342 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([100352, 256], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_25 = async_compile.triton('triton_poi_fused_zeros_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51380224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_25(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ik/cikmzbxnpqr7vars77o32tet27dvc5djs5xj5xr5t4ux5j4nhwwr.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_109, clone_default_112
# Graph fragment:
#   %clone_default_112 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_224,), kwargs = {})
#   %clone_default_109 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_218,), kwargs = {})
triton_poi_fused_26 = async_compile.triton('triton_poi_fused_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 128450560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_26(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sd/csdaupcrhq37d5qygbzforurouhwpi6govjpd7a254fzsiq7ckxl.py
# Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer1_0_bn3 => full_default_18
# Graph fragment:
#   %full_default_18 : [num_users=34] = call_function[target=torch.ops.aten.full.default](args = ([256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_27 = async_compile.triton('triton_poi_fused_zeros_27', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_27(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2p/c2pwlwti4tt4ktomt5y2fgss5acgabbwzyaje763yi72vxudeomc.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_107, clone_default_108, clone_default_110, clone_default_111
# Graph fragment:
#   %clone_default_110 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_220,), kwargs = {})
#   %clone_default_111 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_222,), kwargs = {})
#   %clone_default_107 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_214,), kwargs = {})
#   %clone_default_108 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_216,), kwargs = {})
triton_poi_fused_28 = async_compile.triton('triton_poi_fused_28', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_28(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/nb/cnbzgzp4efgzixzgslocx77g6zf2yksxrp4iwmy7vxaobvxgxnbq.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_203, triton_kernel_wrapper_mutation_204
# Graph fragment:
#   %triton_kernel_wrapper_mutation_204 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 481, constant_args_idx: 679, grid: [(256, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2158, DY: %view_2155, DBETA: %as_strided_default_221, DGAMMA: %as_strided_default_223, M: 100352, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_203 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 482, constant_args_idx: 680, grid: [(256, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2158, DY: %view_2155, INVSTD: %rsqrt_41, GAMMA: %primals_250, DBETA: %as_strided_default_221, DGAMMA: %as_strided_default_223, DX: %permute_168, M: 100352, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_29 = async_compile.triton('triton_poi_fused_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 51380224, 'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_29(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 196*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/lw/clwz7dfjexawfjflv6xum7h2ibkv3thldbqpj3eob3ki73ovbe74.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_122 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_824, torch.float32), kwargs = {})
triton_poi_fused__to_copy_30 = async_compile.triton('triton_poi_fused__to_copy_30', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5898240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ra/cranpj3vvm67xush4bsoklct4h3tdwyd5pfprskx7kcmppdizljm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_106, clone_default_97
# Graph fragment:
#   %clone_default_106 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_212,), kwargs = {})
#   %clone_default_97 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_194,), kwargs = {})
triton_poi_fused_31 = async_compile.triton('triton_poi_fused_31', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 513802240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_31(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ea/cea4wdw3ct5ostiwn3jl2dothejb2itxjhr3iqqk7qszv3ta6j3h.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_230 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_761, %getitem_796), kwargs = {})
#   %mul_380 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, %view_2094), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %getitem_832), kwargs = {})
#   %mul_383 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_231, %view_2229), kwargs = {})
triton_poi_fused_add_mul_32 = async_compile.triton('triton_poi_fused_add_mul_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'in_ptr3': '*bf16', 'in_ptr4': '*i8', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 616562688, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/oq/coqc2kfeil5iguy73d5sjehrvbspmmsph54ep4aq5sw4652xmwc7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_104, clone_default_105, clone_default_95, clone_default_96
# Graph fragment:
#   %clone_default_104 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_208,), kwargs = {})
#   %clone_default_105 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_210,), kwargs = {})
#   %clone_default_95 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_190,), kwargs = {})
#   %clone_default_96 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_192,), kwargs = {})
triton_poi_fused_33 = async_compile.triton('triton_poi_fused_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_33(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/ah/cahwpii6yqqidbwynikbtdua3s4x6airuavrrbkjkmxi56w4ldrb.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_178, triton_kernel_wrapper_mutation_179
# Graph fragment:
#   %triton_kernel_wrapper_mutation_179 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 501, constant_args_idx: 704, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2383, DY: %view_2380, DBETA: %as_strided_default_191, DGAMMA: %as_strided_default_193, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_178 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 502, constant_args_idx: 705, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2383, DY: %view_2380, INVSTD: %rsqrt_36, GAMMA: %primals_220, DBETA: %as_strided_default_191, DGAMMA: %as_strided_default_193, DX: %permute_173, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_34 = async_compile.triton('triton_poi_fused_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 1027604480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_34(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 1024*x2 + 200704*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 196*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 196*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 196*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sf/csfvt66iiidw7dtsckt4bokygzpt7oowhuqzuzhqov5ek4hurzvf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %getitem_868), kwargs = {})
#   %mul_386 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_232, %view_2364), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %getitem_904), kwargs = {})
#   %mul_389 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_233, %view_2499), kwargs = {})
triton_poi_fused_add_mul_35 = async_compile.triton('triton_poi_fused_add_mul_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 616562688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/r3/cr3f4vdeexz73nonzm2ds7oan2y722gkgadbrj5ixczbn5yvab5m.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_129, triton_kernel_wrapper_mutation_130, triton_kernel_wrapper_mutation_133, triton_kernel_wrapper_mutation_134
# Graph fragment:
#   %add_234 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %getitem_940), kwargs = {})
#   %mul_392 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_234, %view_2634), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_392, %getitem_976), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_235, %view_2769), kwargs = {})
#   %triton_kernel_wrapper_mutation_134 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 537, constant_args_idx: 749, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2788, DY: %view_2785, DBETA: %as_strided_default_137, DGAMMA: %as_strided_default_139, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_133 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 538, constant_args_idx: 750, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2788, DY: %view_2785, INVSTD: %rsqrt_27, GAMMA: %primals_166, DBETA: %as_strided_default_137, DGAMMA: %as_strided_default_139, DX: %permute_182, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_130 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 541, constant_args_idx: 753, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2827, DY: %view_2785, DBETA: %full_default_152, DGAMMA: %as_strided_default_135, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_129 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 542, constant_args_idx: 754, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2827, DY: %view_2785, INVSTD: %rsqrt_26, GAMMA: %primals_160, DBETA: %full_default_152, DGAMMA: %as_strided_default_135, DX: %permute_183, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_add_mul_36 = async_compile.triton('triton_poi_fused_add_mul_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1849688064}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/co/ccoi3dp7fsbrpiorwp2tsqplf5tnp4pfgllrj6vf4v65ilgq2x2n.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_67, clone_default_68, clone_default_69
# Graph fragment:
#   %clone_default_68 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_136,), kwargs = {})
#   %clone_default_69 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_138,), kwargs = {})
#   %clone_default_67 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_134,), kwargs = {})
triton_poi_fused_37 = async_compile.triton('triton_poi_fused_37', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 28672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_37(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/3x/c3xr52fmcu3slslyoqrfqxa3zuytmxhqmqqtlefmyij2awlss46d.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_119, triton_kernel_wrapper_mutation_120
# Graph fragment:
#   %triton_kernel_wrapper_mutation_120 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 549, constant_args_idx: 763, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2917, DY: %view_2914, DBETA: %as_strided_default_125, DGAMMA: %as_strided_default_127, M: 401408, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_119 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 550, constant_args_idx: 764, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2917, DY: %view_2914, INVSTD: %rsqrt_24, GAMMA: %primals_148, DBETA: %as_strided_default_125, DGAMMA: %as_strided_default_127, DX: %permute_185, M: 401408, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_38 = async_compile.triton('triton_poi_fused_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_38(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 784*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/vx/cvxujtmrxhvsnzthrdnmw64m2dpvtmotpbhrjaqhnz6lb5jubzsu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_207 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1027, torch.float32), kwargs = {})
triton_poi_fused__to_copy_39 = async_compile.triton('triton_poi_fused__to_copy_39', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1310720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hw/chw3jbdtrlll5bvtrw56yuwkjb7sm44yo6mp6pvuama3pv67fwwu.py
# Topologically Sorted Source Nodes: [full_342], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_342 => full_default_395
# Graph fragment:
#   %full_default_395 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([802816, 256], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_40 = async_compile.triton('triton_poi_fused_zeros_40', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_40(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/cf/ccfiw26jj4fw43atma3vlcih6sjv3cvlgbeyx225ws2rpzyt3vjn.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_61
# Graph fragment:
#   %clone_default_61 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_122,), kwargs = {})
triton_poi_fused_41 = async_compile.triton('triton_poi_fused_41', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 616562688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_41(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/tc/ctcadpghl5ilrojttaigfv4f3s7sop766xqlg4ht326buozcgz6w.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_59, clone_default_60
# Graph fragment:
#   %clone_default_59 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_118,), kwargs = {})
#   %clone_default_60 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_120,), kwargs = {})
triton_poi_fused_42 = async_compile.triton('triton_poi_fused_42', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_42(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mb/cmbhljegmpiyh7caofzidcacks4gfjnjvtwm7zmsr545lnjxef2l.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_114, triton_kernel_wrapper_mutation_115
# Graph fragment:
#   %triton_kernel_wrapper_mutation_115 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 553, constant_args_idx: 768, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2962, DY: %view_2959, DBETA: %as_strided_default_119, DGAMMA: %as_strided_default_121, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_114 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 554, constant_args_idx: 769, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2962, DY: %view_2959, INVSTD: %rsqrt_23, GAMMA: %primals_142, DBETA: %as_strided_default_119, DGAMMA: %as_strided_default_121, DX: %permute_186, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_43 = async_compile.triton('triton_poi_fused_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_43(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 512*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 784*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/cy/ccycb27pgtronwa2erxwl4ymqr5wflqr3w7iyedyie74h6btlrl3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_212 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1039, torch.float32), kwargs = {})
triton_poi_fused__to_copy_44 = async_compile.triton('triton_poi_fused__to_copy_44', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 655360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ej/cej4ptr2af2jiafcfv2l7tgnn3povvs426t5aibjx26vfbvkcahe.py
# Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_bn1 => full_default_64
# Graph fragment:
#   %full_default_64 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_45 = async_compile.triton('triton_poi_fused_zeros_45', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_45(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vw/cvwqfrp2l6gg44axaoua2uhwjkf5usl3kq6ojwdkkqfmywjqmmfu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_53, clone_default_54, clone_default_56, clone_default_57
# Graph fragment:
#   %clone_default_56 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_112,), kwargs = {})
#   %clone_default_57 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_114,), kwargs = {})
#   %clone_default_53 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_106,), kwargs = {})
#   %clone_default_54 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_108,), kwargs = {})
triton_poi_fused_46 = async_compile.triton('triton_poi_fused_46', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_46(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/dm/cdme3jc6j7q7eij7zpeo5ynevljwanfgdvp66sow4oxwfz7uhxhh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_109, triton_kernel_wrapper_mutation_110
# Graph fragment:
#   %triton_kernel_wrapper_mutation_110 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 557, constant_args_idx: 773, grid: [(128, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3007, DY: %view_3004, DBETA: %as_strided_default_113, DGAMMA: %as_strided_default_115, M: 401408, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_109 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 558, constant_args_idx: 774, grid: [(128, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3007, DY: %view_3004, INVSTD: %rsqrt_22, GAMMA: %primals_136, DBETA: %as_strided_default_113, DGAMMA: %as_strided_default_115, DX: %permute_187, M: 401408, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_47 = async_compile.triton('triton_poi_fused_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 102760448, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_47(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 784*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/dj/cdjmrmor77pvwojie2q7ooegyu6zgy5hmqjxjnp3d4ixfpc2k5qo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_217 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1051, torch.float32), kwargs = {})
triton_poi_fused__to_copy_48 = async_compile.triton('triton_poi_fused__to_copy_48', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1474560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4s/c4s72h53zgyvmt7bt6zo7n2ccyjya66pvjdf6mccd2yhhnujgdvh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_43, clone_default_52
# Graph fragment:
#   %clone_default_52 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_104,), kwargs = {})
#   %clone_default_43 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_86,), kwargs = {})
triton_poi_fused_49 = async_compile.triton('triton_poi_fused_49', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1027604480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_49(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/6v/c6vavthhpcxg6pw36wgajwvc5kmp7fock5iv47cud6h3lhzfmwko.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_236 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_988, %getitem_1023), kwargs = {})
#   %mul_398 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_236, %view_2943), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_398, %getitem_1059), kwargs = {})
#   %mul_401 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_237, %view_3078), kwargs = {})
triton_poi_fused_add_mul_50 = async_compile.triton('triton_poi_fused_add_mul_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'in_ptr3': '*bf16', 'in_ptr4': '*i8', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1233125376, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/mr/cmr3ia55tbfil6uxi4gkgs42g46ozw5maxj725u37uagpwh44qyj.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_84, triton_kernel_wrapper_mutation_85
# Graph fragment:
#   %triton_kernel_wrapper_mutation_85 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 577, constant_args_idx: 798, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3232, DY: %view_3229, DBETA: %as_strided_default_83, DGAMMA: %as_strided_default_85, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_84 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 578, constant_args_idx: 799, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3232, DY: %view_3229, INVSTD: %rsqrt_17, GAMMA: %primals_106, DBETA: %as_strided_default_83, DGAMMA: %as_strided_default_85, DX: %permute_192, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_51 = async_compile.triton('triton_poi_fused_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 2055208960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_51(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 512*x2 + 401408*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 784*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 784*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 784*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/55/c552pzczrcsowy2ttgwwfqaws5gcabeq6k5mp5sg7jzwji6smnyd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_65, triton_kernel_wrapper_mutation_66, triton_kernel_wrapper_mutation_69, triton_kernel_wrapper_mutation_70
# Graph fragment:
#   %add_238 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_401, %getitem_1095), kwargs = {})
#   %mul_404 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_238, %view_3213), kwargs = {})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_404, %getitem_1131), kwargs = {})
#   %mul_407 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_239, %view_3348), kwargs = {})
#   %triton_kernel_wrapper_mutation_70 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 589, constant_args_idx: 813, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3367, DY: %view_3364, DBETA: %as_strided_default_65, DGAMMA: %as_strided_default_67, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_69 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 590, constant_args_idx: 814, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3367, DY: %view_3364, INVSTD: %rsqrt_14, GAMMA: %primals_88, DBETA: %as_strided_default_65, DGAMMA: %as_strided_default_67, DX: %permute_195, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_66 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 593, constant_args_idx: 817, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3406, DY: %view_3364, DBETA: %full_default_76, DGAMMA: %as_strided_default_63, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_65 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 594, constant_args_idx: 818, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3406, DY: %view_3364, INVSTD: %rsqrt_13, GAMMA: %primals_82, DBETA: %full_default_76, DGAMMA: %as_strided_default_63, DX: %permute_196, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_add_mul_52 = async_compile.triton('triton_poi_fused_add_mul_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 3699376128}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/2u/c2uy3q5zdlya4dci7a65imvnxp77mtqglerglhx5f5xbjlc2y3rq.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_31, clone_default_32, clone_default_33
# Graph fragment:
#   %clone_default_32 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_64,), kwargs = {})
#   %clone_default_33 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_66,), kwargs = {})
#   %clone_default_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_62,), kwargs = {})
triton_poi_fused_53 = async_compile.triton('triton_poi_fused_53', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_53(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/4j/c4jsunirms27snelhoi4uwuimmssafpdvcma7idl6lrqsimokkja.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_28, clone_default_29, clone_default_30
# Graph fragment:
#   %clone_default_29 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_58,), kwargs = {})
#   %clone_default_30 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_60,), kwargs = {})
#   %clone_default_28 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_56,), kwargs = {})
triton_poi_fused_54 = async_compile.triton('triton_poi_fused_54', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_54(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/pd/cpdnrepmpmt353jykqx6pcpyvqp2lg4yjla6nnxmgukki55c5yeh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_55, triton_kernel_wrapper_mutation_56
# Graph fragment:
#   %triton_kernel_wrapper_mutation_56 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 601, constant_args_idx: 827, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3496, DY: %view_3493, DBETA: %full_default_64, DGAMMA: %as_strided_default_57, M: 1605632, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_55 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 602, constant_args_idx: 828, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3496, DY: %view_3493, INVSTD: %rsqrt_11, GAMMA: %primals_70, DBETA: %full_default_64, DGAMMA: %as_strided_default_57, DX: %permute_198, M: 1605632, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_55 = async_compile.triton('triton_poi_fused_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_55(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp3, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/ye/cye5fksgi5wienifrdq5bwwqfebakv54t6bzlmitvql3ptq655sy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_272 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1182, torch.float32), kwargs = {})
triton_poi_fused__to_copy_56 = async_compile.triton('triton_poi_fused__to_copy_56', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 327680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_56(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ro/croglb333jdraplcrntma2ckgk4ekhcttvr7vwbft752jy35wzce.py
# Topologically Sorted Source Nodes: [full_380], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_380 => full_default_433
# Graph fragment:
#   %full_default_433 : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([1605632, 256], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_57 = async_compile.triton('triton_poi_fused_zeros_57', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_57(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/65/c65rsb2uwkh22naath27jyhubrlnq6o4wqkyv6o2cp7xhsa45iaw.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_27
# Graph fragment:
#   %clone_default_27 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_54,), kwargs = {})
triton_poi_fused_58 = async_compile.triton('triton_poi_fused_58', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1233125376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_58(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jm/cjmdjlfkbrdneef36fy2joi6wld3lkmejxdosrusjkcc52fx6x63.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_50, triton_kernel_wrapper_mutation_51
# Graph fragment:
#   %triton_kernel_wrapper_mutation_51 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 605, constant_args_idx: 832, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3541, DY: %view_3538, DBETA: %as_strided_default_51, DGAMMA: %as_strided_default_53, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_50 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 606, constant_args_idx: 833, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3541, DY: %view_3538, INVSTD: %rsqrt_10, GAMMA: %primals_64, DBETA: %as_strided_default_51, DGAMMA: %as_strided_default_53, DX: %permute_199, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_59 = async_compile.triton('triton_poi_fused_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_59(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0 + 256*x2 + 802816*y1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + 3136*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp5, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/dq/cdqijyjyxebssqtmcinrqh7jpcvpzqwb2ju7mlxsbxjuxw5ih3np.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_277 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1194, torch.float32), kwargs = {})
triton_poi_fused__to_copy_60 = async_compile.triton('triton_poi_fused__to_copy_60', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 163840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_60(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3o/c3olftz5mhhr7pgvmmmbjgcozef5v73yaw3o2p6xzimcp6v2yb3z.py
# Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   bn1 => full_default
# Graph fragment:
#   %full_default : [num_users=16] = call_function[target=torch.ops.aten.full.default](args = ([64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_61 = async_compile.triton('triton_poi_fused_zeros_61', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_61(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/56/c56gwcndkatyqpb4s67fbkdjkyfsri6bokeex6v7tjegsvxig7se.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_19, clone_default_20, clone_default_22, clone_default_23
# Graph fragment:
#   %clone_default_22 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_44,), kwargs = {})
#   %clone_default_23 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_46,), kwargs = {})
#   %clone_default_19 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_38,), kwargs = {})
#   %clone_default_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_40,), kwargs = {})
triton_poi_fused_62 = async_compile.triton('triton_poi_fused_62', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_62(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/7b/c7bf3gzsggqyg35d3eifmkp6wjncxeyx62m3gwswlwtoyhtfxfw6.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_45, triton_kernel_wrapper_mutation_46
# Graph fragment:
#   %triton_kernel_wrapper_mutation_46 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 609, constant_args_idx: 837, grid: [(64, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3586, DY: %view_3583, DBETA: %as_strided_default_45, DGAMMA: %as_strided_default_47, M: 1605632, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_45 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 610, constant_args_idx: 838, grid: [(64, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3586, DY: %view_3583, INVSTD: %rsqrt_9, GAMMA: %primals_58, DBETA: %as_strided_default_45, DGAMMA: %as_strided_default_47, DX: %permute_200, M: 1605632, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_63 = async_compile.triton('triton_poi_fused_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_63(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2 + 3136*y3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x2 + 3136*y3), tmp3, xmask)
    tl.store(out_ptr1 + (x2 + 3136*y3), tmp3, xmask)
''', device_str='cuda')


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/4g/c4g67v3rcqzdg3utzg73qoih3gnujayjommrna3zytbp2mgpeqbj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_282 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1206, torch.float32), kwargs = {})
triton_poi_fused__to_copy_64 = async_compile.triton('triton_poi_fused__to_copy_64', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 368640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_64(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vi/cvi5icemvrfiw24jlllq77x7pwwyifh2h7nx2xj5zgli63d4oyqe.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_18, clone_default_9
# Graph fragment:
#   %clone_default_18 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_36,), kwargs = {})
#   %clone_default_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_18,), kwargs = {})
triton_poi_fused_65 = async_compile.triton('triton_poi_fused_65', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2055208960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_65(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/iz/ciz47nyrozfm2r34iidpnp4doak6xp4hucxnsqltavkybdmaerkl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_240 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1143, %getitem_1178), kwargs = {})
#   %mul_410 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_240, %view_3522), kwargs = {})
#   %add_241 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_410, %getitem_1214), kwargs = {})
#   %mul_413 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_241, %view_3657), kwargs = {})
triton_poi_fused_add_mul_66 = async_compile.triton('triton_poi_fused_add_mul_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'in_ptr3': '*bf16', 'in_ptr4': '*i8', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2466250752, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_66(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/pm/cpmylhzgdq2vk2eho666eitczmw6yqz5yx33uwnbfg2vc5osl73u.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_6, clone_default_7, clone_default_8
# Graph fragment:
#   %clone_default_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_14,), kwargs = {})
#   %clone_default_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_16,), kwargs = {})
#   %clone_default_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_12,), kwargs = {})
triton_poi_fused_67 = async_compile.triton('triton_poi_fused_67', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_67(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/wn/cwnyqzl6c2aal3fy55daj5wbhyzh2irqpf2emhk7vivrlozutkb5.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_16, triton_kernel_wrapper_mutation_17, triton_kernel_wrapper_mutation_20, triton_kernel_wrapper_mutation_21
# Graph fragment:
#   %triton_kernel_wrapper_mutation_21 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 629, constant_args_idx: 862, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3811, DY: %view_3808, DBETA: %as_strided_default_15, DGAMMA: %as_strided_default_17, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_20 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 630, constant_args_idx: 863, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3811, DY: %view_3808, INVSTD: %rsqrt_4, GAMMA: %primals_28, DBETA: %as_strided_default_15, DGAMMA: %as_strided_default_17, DX: %permute_205, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_17 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 633, constant_args_idx: 866, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3850, DY: %view_3808, DBETA: %full_default_18, DGAMMA: %as_strided_default_13, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_16 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 634, constant_args_idx: 867, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3850, DY: %view_3808, INVSTD: %rsqrt_3, GAMMA: %primals_22, DBETA: %full_default_18, DGAMMA: %as_strided_default_13, DX: %permute_206, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_68 = async_compile.triton('triton_poi_fused_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 7398752256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_68(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/qe/cqegc46jrno4ip7cllxsnzbk2ym2bslswgxh2ld4fdupckupvnjx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default, clone_default_1, clone_default_2, clone_default_3, clone_default_4
# Graph fragment:
#   %clone_default_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_6,), kwargs = {})
#   %clone_default_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_8,), kwargs = {})
#   %clone_default_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_2,), kwargs = {})
#   %clone_default_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_4,), kwargs = {})
#   %clone_default : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default,), kwargs = {})
triton_poi_fused_69 = async_compile.triton('triton_poi_fused_69', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_69(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/fl/cflhd23qfd65dkdqgymlmk7blvdb3t7qgjn5em7grtuxg6s6byov.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_322 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1301, torch.float32), kwargs = {})
triton_poi_fused__to_copy_70 = async_compile.triton('triton_poi_fused__to_copy_70', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 40960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_70(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/cj/ccjlgnuvxs3pijc3p4gndjwawvutw323friibb26p45qc47gnx7e.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1262, %getitem_1297), kwargs = {})
triton_poi_fused_add_71 = async_compile.triton('triton_poi_fused_add_71', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_71', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_71(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/tm/ctmx5qjswkqfkxl7nenjrmptzrx3wzg5bymrkymbg2sncvbiweag.py
# Topologically Sorted Source Nodes: [maxpool], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
# Source node to ATen node mapping:
#   maxpool => _low_memory_max_pool_offsets_to_indices
# Graph fragment:
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1262, %getitem_1297), kwargs = {})
#   %_low_memory_max_pool_offsets_to_indices : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_offsets_to_indices.default](args = (%getitem_14, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]), kwargs = {})
#   %max_pool2d_with_indices_backward : [num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%add_243, %getitem_10, [3, 3], [2, 2], [1, 1], [1, 1], False, %_low_memory_max_pool_offsets_to_indices), kwargs = {})
triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_72 = async_compile.triton('triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_72', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8388608, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_72', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_72(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6422528
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


# kernel path: /tmp/torchinductor_yyu496/f6/cf6fht25ofoba6vtiz77ynvlvz7jd77vu7i7xf25mbmtdre5wxl4.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_1, triton_kernel_wrapper_mutation_2
# Graph fragment:
#   %triton_kernel_wrapper_mutation_2 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 645, constant_args_idx: 881, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3985, DY: %view_3982, DBETA: %full_default, DGAMMA: %as_strided_default_1, M: 6422528, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_1 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 646, constant_args_idx: 882, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3985, DY: %view_3982, INVSTD: %rsqrt, GAMMA: %primals_4, DBETA: %full_default, DGAMMA: %as_strided_default_1, DX: %permute_209, M: 6422528, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
triton_poi_fused_73 = async_compile.triton('triton_poi_fused_73', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_73', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4110417920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_73(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:511
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


# Original path: /home/hice1/yyu496/kaggle/CW/ACT5/ops.py:541
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


# kernel path: /tmp/torchinductor_yyu496/om/comm6wk62g46y3coaom6dualufgsmazeiojfe766ikmhr4v6k72p.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_327 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1310, torch.float32), kwargs = {})
triton_poi_fused__to_copy_74 = async_compile.triton('triton_poi_fused__to_copy_74', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_74', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 94080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_74(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, getitem, getitem_1, getitem_2, rsqrt, getitem_7, getitem_8, getitem_9, getitem_10, getitem_12, getitem_14, convert_element_type_2, getitem_15, getitem_16, getitem_17, rsqrt_1, getitem_22, getitem_23, getitem_24, getitem_27, convert_element_type_3, getitem_28, getitem_29, getitem_30, rsqrt_2, getitem_35, getitem_36, getitem_37, getitem_40, convert_element_type_4, getitem_41, getitem_42, getitem_43, rsqrt_3, getitem_48, getitem_49, getitem_50, convert_element_type_5, getitem_51, getitem_52, getitem_53, rsqrt_4, getitem_58, getitem_59, getitem_60, getitem_63, convert_element_type_6, getitem_64, getitem_65, getitem_66, rsqrt_5, getitem_71, getitem_72, getitem_73, getitem_76, convert_element_type_7, getitem_77, getitem_78, getitem_79, rsqrt_6, getitem_84, getitem_85, getitem_86, getitem_89, convert_element_type_8, getitem_90, getitem_91, getitem_92, rsqrt_7, getitem_97, getitem_98, getitem_99, getitem_102, convert_element_type_9, getitem_103, getitem_104, getitem_105, rsqrt_8, getitem_110, getitem_111, getitem_112, getitem_115, convert_element_type_10, getitem_116, getitem_117, getitem_118, rsqrt_9, getitem_123, getitem_124, getitem_125, getitem_128, convert_element_type_11, getitem_129, getitem_130, getitem_131, rsqrt_10, getitem_136, getitem_137, getitem_138, getitem_141, convert_element_type_12, getitem_142, getitem_143, getitem_144, rsqrt_11, getitem_149, getitem_150, getitem_151, getitem_154, convert_element_type_13, getitem_155, getitem_156, getitem_157, rsqrt_12, getitem_162, getitem_163, getitem_164, getitem_167, convert_element_type_14, getitem_168, getitem_169, getitem_170, rsqrt_13, getitem_175, getitem_176, getitem_177, convert_element_type_15, getitem_178, getitem_179, getitem_180, rsqrt_14, getitem_185, getitem_186, getitem_187, getitem_190, convert_element_type_16, getitem_191, getitem_192, getitem_193, rsqrt_15, getitem_198, getitem_199, getitem_200, getitem_203, convert_element_type_17, getitem_204, getitem_205, getitem_206, rsqrt_16, getitem_211, getitem_212, getitem_213, getitem_216, convert_element_type_18, getitem_217, getitem_218, getitem_219, rsqrt_17, getitem_224, getitem_225, getitem_226, getitem_229, convert_element_type_19, getitem_230, getitem_231, getitem_232, rsqrt_18, getitem_237, getitem_238, getitem_239, getitem_242, convert_element_type_20, getitem_243, getitem_244, getitem_245, rsqrt_19, getitem_250, getitem_251, getitem_252, getitem_255, convert_element_type_21, getitem_256, getitem_257, getitem_258, rsqrt_20, getitem_263, getitem_264, getitem_265, getitem_268, convert_element_type_22, getitem_269, getitem_270, getitem_271, rsqrt_21, getitem_276, getitem_277, getitem_278, getitem_281, convert_element_type_23, getitem_282, getitem_283, getitem_284, rsqrt_22, getitem_289, getitem_290, getitem_291, getitem_294, convert_element_type_24, getitem_295, getitem_296, getitem_297, rsqrt_23, getitem_302, getitem_303, getitem_304, getitem_307, convert_element_type_25, getitem_308, getitem_309, getitem_310, rsqrt_24, getitem_315, getitem_316, getitem_317, getitem_320, convert_element_type_26, getitem_321, getitem_322, getitem_323, rsqrt_25, getitem_328, getitem_329, getitem_330, getitem_333, convert_element_type_27, getitem_334, getitem_335, getitem_336, rsqrt_26, getitem_341, getitem_342, getitem_343, convert_element_type_28, getitem_344, getitem_345, getitem_346, rsqrt_27, getitem_351, getitem_352, getitem_353, getitem_356, convert_element_type_29, getitem_357, getitem_358, getitem_359, rsqrt_28, getitem_364, getitem_365, getitem_366, getitem_369, convert_element_type_30, getitem_370, getitem_371, getitem_372, rsqrt_29, getitem_377, getitem_378, getitem_379, getitem_382, convert_element_type_31, getitem_383, getitem_384, getitem_385, rsqrt_30, getitem_390, getitem_391, getitem_392, getitem_395, convert_element_type_32, getitem_396, getitem_397, getitem_398, rsqrt_31, getitem_403, getitem_404, getitem_405, getitem_408, convert_element_type_33, getitem_409, getitem_410, getitem_411, rsqrt_32, getitem_416, getitem_417, getitem_418, getitem_421, convert_element_type_34, getitem_422, getitem_423, getitem_424, rsqrt_33, getitem_429, getitem_430, getitem_431, getitem_434, convert_element_type_35, getitem_435, getitem_436, getitem_437, rsqrt_34, getitem_442, getitem_443, getitem_444, getitem_447, convert_element_type_36, getitem_448, getitem_449, getitem_450, rsqrt_35, getitem_455, getitem_456, getitem_457, getitem_460, convert_element_type_37, getitem_461, getitem_462, getitem_463, rsqrt_36, getitem_468, getitem_469, getitem_470, getitem_473, convert_element_type_38, getitem_474, getitem_475, getitem_476, rsqrt_37, getitem_481, getitem_482, getitem_483, getitem_486, convert_element_type_39, getitem_487, getitem_488, getitem_489, rsqrt_38, getitem_494, getitem_495, getitem_496, getitem_499, convert_element_type_40, getitem_500, getitem_501, getitem_502, rsqrt_39, getitem_507, getitem_508, getitem_509, getitem_512, convert_element_type_41, getitem_513, getitem_514, getitem_515, rsqrt_40, getitem_520, getitem_521, getitem_522, getitem_525, convert_element_type_42, getitem_526, getitem_527, getitem_528, rsqrt_41, getitem_533, getitem_534, getitem_535, getitem_538, convert_element_type_43, getitem_539, getitem_540, getitem_541, rsqrt_42, getitem_546, getitem_547, getitem_548, getitem_551, convert_element_type_44, getitem_552, getitem_553, getitem_554, rsqrt_43, getitem_559, getitem_560, getitem_561, getitem_564, convert_element_type_45, getitem_565, getitem_566, getitem_567, rsqrt_44, getitem_572, getitem_573, getitem_574, getitem_577, convert_element_type_46, getitem_578, getitem_579, getitem_580, rsqrt_45, getitem_585, getitem_586, getitem_587, convert_element_type_47, getitem_588, getitem_589, getitem_590, rsqrt_46, getitem_595, getitem_596, getitem_597, getitem_600, convert_element_type_48, getitem_601, getitem_602, getitem_603, rsqrt_47, getitem_608, getitem_609, getitem_610, getitem_613, convert_element_type_49, getitem_614, getitem_615, getitem_616, rsqrt_48, getitem_621, getitem_622, getitem_623, getitem_626, convert_element_type_50, getitem_627, getitem_628, getitem_629, rsqrt_49, getitem_634, getitem_635, getitem_636, getitem_639, convert_element_type_51, getitem_640, getitem_641, getitem_642, rsqrt_50, getitem_647, getitem_648, getitem_649, getitem_652, convert_element_type_52, getitem_653, getitem_654, getitem_655, rsqrt_51, getitem_660, getitem_661, getitem_662, getitem_665, convert_element_type_53, getitem_666, getitem_667, getitem_668, rsqrt_52, getitem_673, getitem_674, getitem_675, getitem_678, getitem_679, getitem_680, getitem_681, convert_element_type_54, tangents_1 = args
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
    assert_size_stride(getitem, (301056, 16), (16, 1))
    assert_size_stride(getitem_1, (301056, ), (1, ))
    assert_size_stride(getitem_2, (301056, ), (1, ))
    assert_size_stride(rsqrt, (64, ), (1, ))
    assert_size_stride(getitem_7, (1605632, 16), (16, 1))
    assert_size_stride(getitem_8, (1605632, ), (1, ))
    assert_size_stride(getitem_9, (1605632, ), (1, ))
    assert_size_stride(getitem_10, (512, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(getitem_12, (1605632, 8), (8, 1))
    assert_size_stride(getitem_14, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_2, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_15, (401408, 16), (16, 1))
    assert_size_stride(getitem_16, (401408, ), (1, ))
    assert_size_stride(getitem_17, (401408, ), (1, ))
    assert_size_stride(rsqrt_1, (64, ), (1, ))
    assert_size_stride(getitem_22, (401408, 16), (16, 1))
    assert_size_stride(getitem_23, (401408, ), (1, ))
    assert_size_stride(getitem_24, (401408, ), (1, ))
    assert_size_stride(getitem_27, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_3, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(getitem_28, (401408, 16), (16, 1))
    assert_size_stride(getitem_29, (401408, ), (1, ))
    assert_size_stride(getitem_30, (401408, ), (1, ))
    assert_size_stride(rsqrt_2, (64, ), (1, ))
    assert_size_stride(getitem_35, (401408, 16), (16, 1))
    assert_size_stride(getitem_36, (401408, ), (1, ))
    assert_size_stride(getitem_37, (401408, ), (1, ))
    assert_size_stride(getitem_40, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_4, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_41, (401408, 16), (16, 1))
    assert_size_stride(getitem_42, (401408, ), (1, ))
    assert_size_stride(getitem_43, (401408, ), (1, ))
    assert_size_stride(rsqrt_3, (256, ), (1, ))
    assert_size_stride(getitem_48, (1605632, 16), (16, 1))
    assert_size_stride(getitem_49, (1605632, ), (1, ))
    assert_size_stride(getitem_50, (1605632, ), (1, ))
    assert_size_stride(convert_element_type_5, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_51, (401408, 16), (16, 1))
    assert_size_stride(getitem_52, (401408, ), (1, ))
    assert_size_stride(getitem_53, (401408, ), (1, ))
    assert_size_stride(rsqrt_4, (256, ), (1, ))
    assert_size_stride(getitem_58, (1605632, 16), (16, 1))
    assert_size_stride(getitem_59, (1605632, ), (1, ))
    assert_size_stride(getitem_60, (1605632, ), (1, ))
    assert_size_stride(getitem_63, (1605632, 8), (8, 1))
    assert_size_stride(convert_element_type_6, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_64, (1605632, 16), (16, 1))
    assert_size_stride(getitem_65, (1605632, ), (1, ))
    assert_size_stride(getitem_66, (1605632, ), (1, ))
    assert_size_stride(rsqrt_5, (64, ), (1, ))
    assert_size_stride(getitem_71, (401408, 16), (16, 1))
    assert_size_stride(getitem_72, (401408, ), (1, ))
    assert_size_stride(getitem_73, (401408, ), (1, ))
    assert_size_stride(getitem_76, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_7, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(getitem_77, (401408, 16), (16, 1))
    assert_size_stride(getitem_78, (401408, ), (1, ))
    assert_size_stride(getitem_79, (401408, ), (1, ))
    assert_size_stride(rsqrt_6, (64, ), (1, ))
    assert_size_stride(getitem_84, (401408, 16), (16, 1))
    assert_size_stride(getitem_85, (401408, ), (1, ))
    assert_size_stride(getitem_86, (401408, ), (1, ))
    assert_size_stride(getitem_89, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_8, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_90, (401408, 16), (16, 1))
    assert_size_stride(getitem_91, (401408, ), (1, ))
    assert_size_stride(getitem_92, (401408, ), (1, ))
    assert_size_stride(rsqrt_7, (256, ), (1, ))
    assert_size_stride(getitem_97, (1605632, 16), (16, 1))
    assert_size_stride(getitem_98, (1605632, ), (1, ))
    assert_size_stride(getitem_99, (1605632, ), (1, ))
    assert_size_stride(getitem_102, (1605632, 8), (8, 1))
    assert_size_stride(convert_element_type_9, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_103, (1605632, 16), (16, 1))
    assert_size_stride(getitem_104, (1605632, ), (1, ))
    assert_size_stride(getitem_105, (1605632, ), (1, ))
    assert_size_stride(rsqrt_8, (64, ), (1, ))
    assert_size_stride(getitem_110, (401408, 16), (16, 1))
    assert_size_stride(getitem_111, (401408, ), (1, ))
    assert_size_stride(getitem_112, (401408, ), (1, ))
    assert_size_stride(getitem_115, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_10, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(getitem_116, (401408, 16), (16, 1))
    assert_size_stride(getitem_117, (401408, ), (1, ))
    assert_size_stride(getitem_118, (401408, ), (1, ))
    assert_size_stride(rsqrt_9, (64, ), (1, ))
    assert_size_stride(getitem_123, (401408, 16), (16, 1))
    assert_size_stride(getitem_124, (401408, ), (1, ))
    assert_size_stride(getitem_125, (401408, ), (1, ))
    assert_size_stride(getitem_128, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_11, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_129, (401408, 16), (16, 1))
    assert_size_stride(getitem_130, (401408, ), (1, ))
    assert_size_stride(getitem_131, (401408, ), (1, ))
    assert_size_stride(rsqrt_10, (256, ), (1, ))
    assert_size_stride(getitem_136, (1605632, 16), (16, 1))
    assert_size_stride(getitem_137, (1605632, ), (1, ))
    assert_size_stride(getitem_138, (1605632, ), (1, ))
    assert_size_stride(getitem_141, (1605632, 8), (8, 1))
    assert_size_stride(convert_element_type_12, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_142, (1605632, 16), (16, 1))
    assert_size_stride(getitem_143, (1605632, ), (1, ))
    assert_size_stride(getitem_144, (1605632, ), (1, ))
    assert_size_stride(rsqrt_11, (128, ), (1, ))
    assert_size_stride(getitem_149, (802816, 16), (16, 1))
    assert_size_stride(getitem_150, (802816, ), (1, ))
    assert_size_stride(getitem_151, (802816, ), (1, ))
    assert_size_stride(getitem_154, (802816, 8), (8, 1))
    assert_size_stride(convert_element_type_13, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(getitem_155, (802816, 16), (16, 1))
    assert_size_stride(getitem_156, (802816, ), (1, ))
    assert_size_stride(getitem_157, (802816, ), (1, ))
    assert_size_stride(rsqrt_12, (128, ), (1, ))
    assert_size_stride(getitem_162, (200704, 16), (16, 1))
    assert_size_stride(getitem_163, (200704, ), (1, ))
    assert_size_stride(getitem_164, (200704, ), (1, ))
    assert_size_stride(getitem_167, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_14, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(getitem_168, (200704, 16), (16, 1))
    assert_size_stride(getitem_169, (200704, ), (1, ))
    assert_size_stride(getitem_170, (200704, ), (1, ))
    assert_size_stride(rsqrt_13, (512, ), (1, ))
    assert_size_stride(getitem_175, (802816, 16), (16, 1))
    assert_size_stride(getitem_176, (802816, ), (1, ))
    assert_size_stride(getitem_177, (802816, ), (1, ))
    assert_size_stride(convert_element_type_15, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_178, (1605632, 16), (16, 1))
    assert_size_stride(getitem_179, (1605632, ), (1, ))
    assert_size_stride(getitem_180, (1605632, ), (1, ))
    assert_size_stride(rsqrt_14, (512, ), (1, ))
    assert_size_stride(getitem_185, (802816, 16), (16, 1))
    assert_size_stride(getitem_186, (802816, ), (1, ))
    assert_size_stride(getitem_187, (802816, ), (1, ))
    assert_size_stride(getitem_190, (802816, 8), (8, 1))
    assert_size_stride(convert_element_type_16, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_191, (802816, 16), (16, 1))
    assert_size_stride(getitem_192, (802816, ), (1, ))
    assert_size_stride(getitem_193, (802816, ), (1, ))
    assert_size_stride(rsqrt_15, (128, ), (1, ))
    assert_size_stride(getitem_198, (200704, 16), (16, 1))
    assert_size_stride(getitem_199, (200704, ), (1, ))
    assert_size_stride(getitem_200, (200704, ), (1, ))
    assert_size_stride(getitem_203, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_17, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(getitem_204, (200704, 16), (16, 1))
    assert_size_stride(getitem_205, (200704, ), (1, ))
    assert_size_stride(getitem_206, (200704, ), (1, ))
    assert_size_stride(rsqrt_16, (128, ), (1, ))
    assert_size_stride(getitem_211, (200704, 16), (16, 1))
    assert_size_stride(getitem_212, (200704, ), (1, ))
    assert_size_stride(getitem_213, (200704, ), (1, ))
    assert_size_stride(getitem_216, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_18, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(getitem_217, (200704, 16), (16, 1))
    assert_size_stride(getitem_218, (200704, ), (1, ))
    assert_size_stride(getitem_219, (200704, ), (1, ))
    assert_size_stride(rsqrt_17, (512, ), (1, ))
    assert_size_stride(getitem_224, (802816, 16), (16, 1))
    assert_size_stride(getitem_225, (802816, ), (1, ))
    assert_size_stride(getitem_226, (802816, ), (1, ))
    assert_size_stride(getitem_229, (802816, 8), (8, 1))
    assert_size_stride(convert_element_type_19, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_230, (802816, 16), (16, 1))
    assert_size_stride(getitem_231, (802816, ), (1, ))
    assert_size_stride(getitem_232, (802816, ), (1, ))
    assert_size_stride(rsqrt_18, (128, ), (1, ))
    assert_size_stride(getitem_237, (200704, 16), (16, 1))
    assert_size_stride(getitem_238, (200704, ), (1, ))
    assert_size_stride(getitem_239, (200704, ), (1, ))
    assert_size_stride(getitem_242, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_20, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(getitem_243, (200704, 16), (16, 1))
    assert_size_stride(getitem_244, (200704, ), (1, ))
    assert_size_stride(getitem_245, (200704, ), (1, ))
    assert_size_stride(rsqrt_19, (128, ), (1, ))
    assert_size_stride(getitem_250, (200704, 16), (16, 1))
    assert_size_stride(getitem_251, (200704, ), (1, ))
    assert_size_stride(getitem_252, (200704, ), (1, ))
    assert_size_stride(getitem_255, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_21, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(getitem_256, (200704, 16), (16, 1))
    assert_size_stride(getitem_257, (200704, ), (1, ))
    assert_size_stride(getitem_258, (200704, ), (1, ))
    assert_size_stride(rsqrt_20, (512, ), (1, ))
    assert_size_stride(getitem_263, (802816, 16), (16, 1))
    assert_size_stride(getitem_264, (802816, ), (1, ))
    assert_size_stride(getitem_265, (802816, ), (1, ))
    assert_size_stride(getitem_268, (802816, 8), (8, 1))
    assert_size_stride(convert_element_type_22, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_269, (802816, 16), (16, 1))
    assert_size_stride(getitem_270, (802816, ), (1, ))
    assert_size_stride(getitem_271, (802816, ), (1, ))
    assert_size_stride(rsqrt_21, (128, ), (1, ))
    assert_size_stride(getitem_276, (200704, 16), (16, 1))
    assert_size_stride(getitem_277, (200704, ), (1, ))
    assert_size_stride(getitem_278, (200704, ), (1, ))
    assert_size_stride(getitem_281, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_23, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(getitem_282, (200704, 16), (16, 1))
    assert_size_stride(getitem_283, (200704, ), (1, ))
    assert_size_stride(getitem_284, (200704, ), (1, ))
    assert_size_stride(rsqrt_22, (128, ), (1, ))
    assert_size_stride(getitem_289, (200704, 16), (16, 1))
    assert_size_stride(getitem_290, (200704, ), (1, ))
    assert_size_stride(getitem_291, (200704, ), (1, ))
    assert_size_stride(getitem_294, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_24, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(getitem_295, (200704, 16), (16, 1))
    assert_size_stride(getitem_296, (200704, ), (1, ))
    assert_size_stride(getitem_297, (200704, ), (1, ))
    assert_size_stride(rsqrt_23, (512, ), (1, ))
    assert_size_stride(getitem_302, (802816, 16), (16, 1))
    assert_size_stride(getitem_303, (802816, ), (1, ))
    assert_size_stride(getitem_304, (802816, ), (1, ))
    assert_size_stride(getitem_307, (802816, 8), (8, 1))
    assert_size_stride(convert_element_type_25, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_308, (802816, 16), (16, 1))
    assert_size_stride(getitem_309, (802816, ), (1, ))
    assert_size_stride(getitem_310, (802816, ), (1, ))
    assert_size_stride(rsqrt_24, (256, ), (1, ))
    assert_size_stride(getitem_315, (401408, 16), (16, 1))
    assert_size_stride(getitem_316, (401408, ), (1, ))
    assert_size_stride(getitem_317, (401408, ), (1, ))
    assert_size_stride(getitem_320, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_26, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_321, (401408, 16), (16, 1))
    assert_size_stride(getitem_322, (401408, ), (1, ))
    assert_size_stride(getitem_323, (401408, ), (1, ))
    assert_size_stride(rsqrt_25, (256, ), (1, ))
    assert_size_stride(getitem_328, (100352, 16), (16, 1))
    assert_size_stride(getitem_329, (100352, ), (1, ))
    assert_size_stride(getitem_330, (100352, ), (1, ))
    assert_size_stride(getitem_333, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_27, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_334, (100352, 16), (16, 1))
    assert_size_stride(getitem_335, (100352, ), (1, ))
    assert_size_stride(getitem_336, (100352, ), (1, ))
    assert_size_stride(rsqrt_26, (1024, ), (1, ))
    assert_size_stride(getitem_341, (401408, 16), (16, 1))
    assert_size_stride(getitem_342, (401408, ), (1, ))
    assert_size_stride(getitem_343, (401408, ), (1, ))
    assert_size_stride(convert_element_type_28, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_344, (802816, 16), (16, 1))
    assert_size_stride(getitem_345, (802816, ), (1, ))
    assert_size_stride(getitem_346, (802816, ), (1, ))
    assert_size_stride(rsqrt_27, (1024, ), (1, ))
    assert_size_stride(getitem_351, (401408, 16), (16, 1))
    assert_size_stride(getitem_352, (401408, ), (1, ))
    assert_size_stride(getitem_353, (401408, ), (1, ))
    assert_size_stride(getitem_356, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_29, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_357, (401408, 16), (16, 1))
    assert_size_stride(getitem_358, (401408, ), (1, ))
    assert_size_stride(getitem_359, (401408, ), (1, ))
    assert_size_stride(rsqrt_28, (256, ), (1, ))
    assert_size_stride(getitem_364, (100352, 16), (16, 1))
    assert_size_stride(getitem_365, (100352, ), (1, ))
    assert_size_stride(getitem_366, (100352, ), (1, ))
    assert_size_stride(getitem_369, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_30, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_370, (100352, 16), (16, 1))
    assert_size_stride(getitem_371, (100352, ), (1, ))
    assert_size_stride(getitem_372, (100352, ), (1, ))
    assert_size_stride(rsqrt_29, (256, ), (1, ))
    assert_size_stride(getitem_377, (100352, 16), (16, 1))
    assert_size_stride(getitem_378, (100352, ), (1, ))
    assert_size_stride(getitem_379, (100352, ), (1, ))
    assert_size_stride(getitem_382, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_31, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_383, (100352, 16), (16, 1))
    assert_size_stride(getitem_384, (100352, ), (1, ))
    assert_size_stride(getitem_385, (100352, ), (1, ))
    assert_size_stride(rsqrt_30, (1024, ), (1, ))
    assert_size_stride(getitem_390, (401408, 16), (16, 1))
    assert_size_stride(getitem_391, (401408, ), (1, ))
    assert_size_stride(getitem_392, (401408, ), (1, ))
    assert_size_stride(getitem_395, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_32, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_396, (401408, 16), (16, 1))
    assert_size_stride(getitem_397, (401408, ), (1, ))
    assert_size_stride(getitem_398, (401408, ), (1, ))
    assert_size_stride(rsqrt_31, (256, ), (1, ))
    assert_size_stride(getitem_403, (100352, 16), (16, 1))
    assert_size_stride(getitem_404, (100352, ), (1, ))
    assert_size_stride(getitem_405, (100352, ), (1, ))
    assert_size_stride(getitem_408, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_33, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_409, (100352, 16), (16, 1))
    assert_size_stride(getitem_410, (100352, ), (1, ))
    assert_size_stride(getitem_411, (100352, ), (1, ))
    assert_size_stride(rsqrt_32, (256, ), (1, ))
    assert_size_stride(getitem_416, (100352, 16), (16, 1))
    assert_size_stride(getitem_417, (100352, ), (1, ))
    assert_size_stride(getitem_418, (100352, ), (1, ))
    assert_size_stride(getitem_421, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_34, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_422, (100352, 16), (16, 1))
    assert_size_stride(getitem_423, (100352, ), (1, ))
    assert_size_stride(getitem_424, (100352, ), (1, ))
    assert_size_stride(rsqrt_33, (1024, ), (1, ))
    assert_size_stride(getitem_429, (401408, 16), (16, 1))
    assert_size_stride(getitem_430, (401408, ), (1, ))
    assert_size_stride(getitem_431, (401408, ), (1, ))
    assert_size_stride(getitem_434, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_35, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_435, (401408, 16), (16, 1))
    assert_size_stride(getitem_436, (401408, ), (1, ))
    assert_size_stride(getitem_437, (401408, ), (1, ))
    assert_size_stride(rsqrt_34, (256, ), (1, ))
    assert_size_stride(getitem_442, (100352, 16), (16, 1))
    assert_size_stride(getitem_443, (100352, ), (1, ))
    assert_size_stride(getitem_444, (100352, ), (1, ))
    assert_size_stride(getitem_447, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_36, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_448, (100352, 16), (16, 1))
    assert_size_stride(getitem_449, (100352, ), (1, ))
    assert_size_stride(getitem_450, (100352, ), (1, ))
    assert_size_stride(rsqrt_35, (256, ), (1, ))
    assert_size_stride(getitem_455, (100352, 16), (16, 1))
    assert_size_stride(getitem_456, (100352, ), (1, ))
    assert_size_stride(getitem_457, (100352, ), (1, ))
    assert_size_stride(getitem_460, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_37, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_461, (100352, 16), (16, 1))
    assert_size_stride(getitem_462, (100352, ), (1, ))
    assert_size_stride(getitem_463, (100352, ), (1, ))
    assert_size_stride(rsqrt_36, (1024, ), (1, ))
    assert_size_stride(getitem_468, (401408, 16), (16, 1))
    assert_size_stride(getitem_469, (401408, ), (1, ))
    assert_size_stride(getitem_470, (401408, ), (1, ))
    assert_size_stride(getitem_473, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_38, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_474, (401408, 16), (16, 1))
    assert_size_stride(getitem_475, (401408, ), (1, ))
    assert_size_stride(getitem_476, (401408, ), (1, ))
    assert_size_stride(rsqrt_37, (256, ), (1, ))
    assert_size_stride(getitem_481, (100352, 16), (16, 1))
    assert_size_stride(getitem_482, (100352, ), (1, ))
    assert_size_stride(getitem_483, (100352, ), (1, ))
    assert_size_stride(getitem_486, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_39, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_487, (100352, 16), (16, 1))
    assert_size_stride(getitem_488, (100352, ), (1, ))
    assert_size_stride(getitem_489, (100352, ), (1, ))
    assert_size_stride(rsqrt_38, (256, ), (1, ))
    assert_size_stride(getitem_494, (100352, 16), (16, 1))
    assert_size_stride(getitem_495, (100352, ), (1, ))
    assert_size_stride(getitem_496, (100352, ), (1, ))
    assert_size_stride(getitem_499, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_40, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_500, (100352, 16), (16, 1))
    assert_size_stride(getitem_501, (100352, ), (1, ))
    assert_size_stride(getitem_502, (100352, ), (1, ))
    assert_size_stride(rsqrt_39, (1024, ), (1, ))
    assert_size_stride(getitem_507, (401408, 16), (16, 1))
    assert_size_stride(getitem_508, (401408, ), (1, ))
    assert_size_stride(getitem_509, (401408, ), (1, ))
    assert_size_stride(getitem_512, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_41, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_513, (401408, 16), (16, 1))
    assert_size_stride(getitem_514, (401408, ), (1, ))
    assert_size_stride(getitem_515, (401408, ), (1, ))
    assert_size_stride(rsqrt_40, (256, ), (1, ))
    assert_size_stride(getitem_520, (100352, 16), (16, 1))
    assert_size_stride(getitem_521, (100352, ), (1, ))
    assert_size_stride(getitem_522, (100352, ), (1, ))
    assert_size_stride(getitem_525, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_42, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_526, (100352, 16), (16, 1))
    assert_size_stride(getitem_527, (100352, ), (1, ))
    assert_size_stride(getitem_528, (100352, ), (1, ))
    assert_size_stride(rsqrt_41, (256, ), (1, ))
    assert_size_stride(getitem_533, (100352, 16), (16, 1))
    assert_size_stride(getitem_534, (100352, ), (1, ))
    assert_size_stride(getitem_535, (100352, ), (1, ))
    assert_size_stride(getitem_538, (100352, 8), (8, 1))
    assert_size_stride(convert_element_type_43, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_539, (100352, 16), (16, 1))
    assert_size_stride(getitem_540, (100352, ), (1, ))
    assert_size_stride(getitem_541, (100352, ), (1, ))
    assert_size_stride(rsqrt_42, (1024, ), (1, ))
    assert_size_stride(getitem_546, (401408, 16), (16, 1))
    assert_size_stride(getitem_547, (401408, ), (1, ))
    assert_size_stride(getitem_548, (401408, ), (1, ))
    assert_size_stride(getitem_551, (401408, 8), (8, 1))
    assert_size_stride(convert_element_type_44, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_552, (401408, 16), (16, 1))
    assert_size_stride(getitem_553, (401408, ), (1, ))
    assert_size_stride(getitem_554, (401408, ), (1, ))
    assert_size_stride(rsqrt_43, (512, ), (1, ))
    assert_size_stride(getitem_559, (200704, 16), (16, 1))
    assert_size_stride(getitem_560, (200704, ), (1, ))
    assert_size_stride(getitem_561, (200704, ), (1, ))
    assert_size_stride(getitem_564, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_45, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(getitem_565, (200704, 16), (16, 1))
    assert_size_stride(getitem_566, (200704, ), (1, ))
    assert_size_stride(getitem_567, (200704, ), (1, ))
    assert_size_stride(rsqrt_44, (512, ), (1, ))
    assert_size_stride(getitem_572, (50176, 16), (16, 1))
    assert_size_stride(getitem_573, (50176, ), (1, ))
    assert_size_stride(getitem_574, (50176, ), (1, ))
    assert_size_stride(getitem_577, (50176, 8), (8, 1))
    assert_size_stride(convert_element_type_46, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_578, (50176, 16), (16, 1))
    assert_size_stride(getitem_579, (50176, ), (1, ))
    assert_size_stride(getitem_580, (50176, ), (1, ))
    assert_size_stride(rsqrt_45, (2048, ), (1, ))
    assert_size_stride(getitem_585, (200704, 16), (16, 1))
    assert_size_stride(getitem_586, (200704, ), (1, ))
    assert_size_stride(getitem_587, (200704, ), (1, ))
    assert_size_stride(convert_element_type_47, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_588, (401408, 16), (16, 1))
    assert_size_stride(getitem_589, (401408, ), (1, ))
    assert_size_stride(getitem_590, (401408, ), (1, ))
    assert_size_stride(rsqrt_46, (2048, ), (1, ))
    assert_size_stride(getitem_595, (200704, 16), (16, 1))
    assert_size_stride(getitem_596, (200704, ), (1, ))
    assert_size_stride(getitem_597, (200704, ), (1, ))
    assert_size_stride(getitem_600, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_48, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(getitem_601, (200704, 16), (16, 1))
    assert_size_stride(getitem_602, (200704, ), (1, ))
    assert_size_stride(getitem_603, (200704, ), (1, ))
    assert_size_stride(rsqrt_47, (512, ), (1, ))
    assert_size_stride(getitem_608, (50176, 16), (16, 1))
    assert_size_stride(getitem_609, (50176, ), (1, ))
    assert_size_stride(getitem_610, (50176, ), (1, ))
    assert_size_stride(getitem_613, (50176, 8), (8, 1))
    assert_size_stride(convert_element_type_49, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(getitem_614, (50176, 16), (16, 1))
    assert_size_stride(getitem_615, (50176, ), (1, ))
    assert_size_stride(getitem_616, (50176, ), (1, ))
    assert_size_stride(rsqrt_48, (512, ), (1, ))
    assert_size_stride(getitem_621, (50176, 16), (16, 1))
    assert_size_stride(getitem_622, (50176, ), (1, ))
    assert_size_stride(getitem_623, (50176, ), (1, ))
    assert_size_stride(getitem_626, (50176, 8), (8, 1))
    assert_size_stride(convert_element_type_50, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_627, (50176, 16), (16, 1))
    assert_size_stride(getitem_628, (50176, ), (1, ))
    assert_size_stride(getitem_629, (50176, ), (1, ))
    assert_size_stride(rsqrt_49, (2048, ), (1, ))
    assert_size_stride(getitem_634, (200704, 16), (16, 1))
    assert_size_stride(getitem_635, (200704, ), (1, ))
    assert_size_stride(getitem_636, (200704, ), (1, ))
    assert_size_stride(getitem_639, (200704, 8), (8, 1))
    assert_size_stride(convert_element_type_51, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(getitem_640, (200704, 16), (16, 1))
    assert_size_stride(getitem_641, (200704, ), (1, ))
    assert_size_stride(getitem_642, (200704, ), (1, ))
    assert_size_stride(rsqrt_50, (512, ), (1, ))
    assert_size_stride(getitem_647, (50176, 16), (16, 1))
    assert_size_stride(getitem_648, (50176, ), (1, ))
    assert_size_stride(getitem_649, (50176, ), (1, ))
    assert_size_stride(getitem_652, (50176, 8), (8, 1))
    assert_size_stride(convert_element_type_52, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(getitem_653, (50176, 16), (16, 1))
    assert_size_stride(getitem_654, (50176, ), (1, ))
    assert_size_stride(getitem_655, (50176, ), (1, ))
    assert_size_stride(rsqrt_51, (512, ), (1, ))
    assert_size_stride(getitem_660, (50176, 16), (16, 1))
    assert_size_stride(getitem_661, (50176, ), (1, ))
    assert_size_stride(getitem_662, (50176, ), (1, ))
    assert_size_stride(getitem_665, (50176, 8), (8, 1))
    assert_size_stride(convert_element_type_53, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_666, (50176, 16), (16, 1))
    assert_size_stride(getitem_667, (50176, ), (1, ))
    assert_size_stride(getitem_668, (50176, ), (1, ))
    assert_size_stride(rsqrt_52, (2048, ), (1, ))
    assert_size_stride(getitem_673, (200704, 16), (16, 1))
    assert_size_stride(getitem_674, (200704, ), (1, ))
    assert_size_stride(getitem_675, (200704, ), (1, ))
    assert_size_stride(getitem_678, (200704, 8), (8, 1))
    assert_size_stride(getitem_679, (4096, 16), (16, 1))
    assert_size_stride(getitem_680, (4096, ), (1, ))
    assert_size_stride(getitem_681, (4096, ), (1, ))
    assert_size_stride(convert_element_type_54, (100, 2048), (2048, 1))
    assert_size_stride(tangents_1, (512, 100), (100, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_679, getitem_680, getitem_681, buf0, 16, 1, 256, 1, 2, 16, 16, 4096, 1, 1, stream=stream0)
        del getitem_679
        del getitem_680
        del getitem_681
        buf2 = empty_strided_cuda((100, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (100, 512), (1, 100), 0), reinterpret_tensor(buf0, (512, 2048), (2048, 1), 0), out=buf2)
        buf3 = reinterpret_tensor(buf0, (512, 2048), (2048, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [view_1643], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, convert_element_type_54, out=buf3)
        del convert_element_type_54
        del tangents_1
        buf4 = empty_strided_cuda((100, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(buf2, buf4, 204800, stream=stream0)
        del buf2
        buf5 = empty_strided_cuda((200704, 256), (256, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_257], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_1.run(buf5, 51380224, stream=stream0)
        buf6 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf72 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf5, buf6, buf72, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_678, reinterpret_tensor(buf6, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_678
        buf8 = empty_strided_cuda((200704, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_673, getitem_674, getitem_675, buf8, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_673
        del getitem_674
        del getitem_675
        buf10 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_3.run(buf10, 2048, stream=stream0)
        buf11 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf12 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf76 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf77 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(buf10, buf11, buf12, buf76, buf77, 2048, stream=stream0)
        buf13 = empty_strided_cuda((512, 2048, 49), (100352, 49, 1), torch.bfloat16)
        buf17 = empty_strided_cuda((512, 2048, 49), (100352, 49, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(buf3, buf6, buf13, buf17, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf8, (512, 2048, 49), (100352, 49, 1), 0), buf13, buf11, buf12, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf16 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf8, (512, 2048, 49), (100352, 49, 1), 0), buf17, rsqrt_52, primals_316, buf11, buf12, buf16, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del primals_316
        del rsqrt_52
        buf19 = empty_strided_cuda((50176, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_666, getitem_667, getitem_668, buf19, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_666
        del getitem_667
        del getitem_668
        buf21 = empty_strided_cuda((1, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf22 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf16, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf21, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_53
        buf23 = buf22[0]
        assert_size_stride(buf23, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf23, 16, 'torch.ops.aten.convolution_backward.default')
        del buf22
        buf24 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf25 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf16, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf19, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf24, (2048, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf26 = buf25[1]
        assert_size_stride(buf26, (2048, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf26, 16, 'torch.ops.aten.convolution_backward.default')
        del buf25
        buf27 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf26, buf27, 1048576, stream=stream0)
        del buf26
        buf28 = empty_strided_cuda((50176, 256), (256, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_260], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_7.run(buf28, 12845056, stream=stream0)
        buf29 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        buf51 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(buf28, buf29, buf51, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_665, reinterpret_tensor(buf29, (50176, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 50176, 1, 1, stream=stream0)
        del getitem_665
        buf31 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_660, getitem_661, getitem_662, buf31, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_660
        del getitem_661
        del getitem_662
        buf33 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_9.run(buf33, 512, stream=stream0)
        buf34 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf35 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf55 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf56 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(buf33, buf34, buf35, buf55, buf56, 512, stream=stream0)
        buf36 = empty_strided_cuda((512, 512, 49), (25088, 49, 1), torch.bfloat16)
        buf40 = empty_strided_cuda((512, 512, 49), (25088, 49, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf23, buf29, buf36, buf40, 262144, 49, stream=stream0)
        del buf23
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf31, (512, 512, 49), (25088, 49, 1), 0), buf36, buf34, buf35, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf39 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf31, (512, 512, 49), (25088, 49, 1), 0), buf40, rsqrt_51, primals_310, buf34, buf35, buf39, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del primals_310
        del rsqrt_51
        buf42 = reinterpret_tensor(buf40, (50176, 256), (256, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_653, getitem_654, getitem_655, buf42, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_653
        del getitem_654
        del getitem_655
        buf44 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf45 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf39, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf44, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_52
        buf46 = buf45[0]
        assert_size_stride(buf46, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf46, 16, 'torch.ops.aten.convolution_backward.default')
        del buf45
        buf47 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf48 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf39, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf42, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf47, (512, 512, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf49 = buf48[1]
        assert_size_stride(buf49, (512, 512, 3, 3), (4608, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf49, 16, 'torch.ops.aten.convolution_backward.default')
        del buf48
        buf50 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf49, buf50, 2359296, stream=stream0)
        del buf49
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_652, reinterpret_tensor(buf51, (50176, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 50176, 1, 1, stream=stream0)
        del getitem_652
        buf53 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_647, getitem_648, getitem_649, buf53, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_647
        del getitem_648
        del getitem_649
        buf57 = buf39; del buf39  # reuse
        buf61 = reinterpret_tensor(buf31, (512, 512, 49), (25088, 49, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf46, buf51, buf57, buf61, 262144, 49, stream=stream0)
        del buf46
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf53, (512, 512, 49), (25088, 49, 1), 0), buf57, buf55, buf56, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf60 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf53, (512, 512, 49), (25088, 49, 1), 0), buf61, rsqrt_50, primals_304, buf55, buf56, buf60, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del primals_304
        del rsqrt_50
        buf63 = reinterpret_tensor(buf16, (200704, 256), (256, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_640, getitem_641, getitem_642, buf63, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_640
        del getitem_641
        del getitem_642
        buf65 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf66 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf60, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf65, (512, 2048, 7, 7), (0, 0, 0, 0), 0), convert_element_type_51, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_51
        buf67 = buf66[0]
        assert_size_stride(buf67, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf67, 16, 'torch.ops.aten.convolution_backward.default')
        del buf66
        buf68 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf69 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf60, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf63, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf68, (512, 2048, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf70 = buf69[1]
        assert_size_stride(buf70, (512, 2048, 1, 1), (2048, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf70, 16, 'torch.ops.aten.convolution_backward.default')
        del buf69
        buf71 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf70, buf71, 1048576, stream=stream0)
        del buf70
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_639, reinterpret_tensor(buf72, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_639
        buf74 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_634, getitem_635, getitem_636, buf74, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_634
        del getitem_635
        del getitem_636
        buf78 = reinterpret_tensor(buf8, (512, 2048, 49), (100352, 49, 1), 0); del buf8  # reuse
        buf82 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_13.run(buf3, buf6, buf67, buf72, buf78, buf82, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf74, (512, 2048, 49), (100352, 49, 1), 0), buf78, buf76, buf77, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf81 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf74, (512, 2048, 49), (100352, 49, 1), 0), buf82, rsqrt_49, primals_298, buf76, buf77, buf81, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del buf74
        del primals_298
        del rsqrt_49
        buf84 = reinterpret_tensor(buf60, (50176, 256), (256, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_627, getitem_628, getitem_629, buf84, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_627
        del getitem_628
        del getitem_629
        buf86 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf87 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf81, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf86, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_50, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_50
        buf88 = buf87[0]
        assert_size_stride(buf88, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf88, 16, 'torch.ops.aten.convolution_backward.default')
        del buf87
        buf89 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf90 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf81, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf84, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf89, (2048, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf91 = buf90[1]
        assert_size_stride(buf91, (2048, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf91, 16, 'torch.ops.aten.convolution_backward.default')
        del buf90
        buf92 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf91, buf92, 1048576, stream=stream0)
        del buf91
        buf93 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        buf114 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(buf28, buf93, buf114, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_626, reinterpret_tensor(buf93, (50176, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 50176, 1, 1, stream=stream0)
        del getitem_626
        buf95 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_621, getitem_622, getitem_623, buf95, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_621
        del getitem_622
        del getitem_623
        buf97 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf98 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf118 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf119 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(buf33, buf97, buf98, buf118, buf119, 512, stream=stream0)
        buf99 = buf61; del buf61  # reuse
        buf103 = reinterpret_tensor(buf53, (512, 512, 49), (25088, 49, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf88, buf93, buf99, buf103, 262144, 49, stream=stream0)
        del buf88
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf95, (512, 512, 49), (25088, 49, 1), 0), buf99, buf97, buf98, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf102 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf95, (512, 512, 49), (25088, 49, 1), 0), buf103, rsqrt_48, primals_292, buf97, buf98, buf102, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del primals_292
        del rsqrt_48
        buf105 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_614, getitem_615, getitem_616, buf105, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_614
        del getitem_615
        del getitem_616
        buf107 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf108 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf102, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf107, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_49
        buf109 = buf108[0]
        assert_size_stride(buf109, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf109, 16, 'torch.ops.aten.convolution_backward.default')
        del buf108
        buf110 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf111 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf102, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf105, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf110, (512, 512, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf112 = buf111[1]
        assert_size_stride(buf112, (512, 512, 3, 3), (4608, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf112, 16, 'torch.ops.aten.convolution_backward.default')
        del buf111
        buf113 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf112, buf113, 2359296, stream=stream0)
        del buf112
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_613, reinterpret_tensor(buf114, (50176, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 50176, 1, 1, stream=stream0)
        del getitem_613
        buf116 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_608, getitem_609, getitem_610, buf116, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_608
        del getitem_609
        del getitem_610
        buf120 = buf102; del buf102  # reuse
        buf124 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf109, buf114, buf120, buf124, 262144, 49, stream=stream0)
        del buf109
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf116, (512, 512, 49), (25088, 49, 1), 0), buf120, buf118, buf119, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf123 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf116, (512, 512, 49), (25088, 49, 1), 0), buf124, rsqrt_47, primals_286, buf118, buf119, buf123, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del primals_286
        del rsqrt_47
        buf126 = reinterpret_tensor(buf81, (200704, 256), (256, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_601, getitem_602, getitem_603, buf126, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_601
        del getitem_602
        del getitem_603
        buf128 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf129 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf123, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf128, (512, 2048, 7, 7), (0, 0, 0, 0), 0), convert_element_type_48, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_48
        buf130 = buf129[0]
        assert_size_stride(buf130, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf130, 16, 'torch.ops.aten.convolution_backward.default')
        del buf129
        buf131 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf132 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf123, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf126, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf131, (512, 2048, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf133 = buf132[1]
        assert_size_stride(buf133, (512, 2048, 1, 1), (2048, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf133, 16, 'torch.ops.aten.convolution_backward.default')
        del buf132
        buf134 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf133, buf134, 1048576, stream=stream0)
        del buf133
        buf135 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf191 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf5, buf135, buf191, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_600, reinterpret_tensor(buf135, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_600
        buf137 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_595, getitem_596, getitem_597, buf137, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_595
        del getitem_596
        del getitem_597
        buf139 = reinterpret_tensor(buf82, (512, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_14.run(buf3, buf6, buf67, buf72, buf130, buf135, buf139, 1048576, 49, stream=stream0)
        del buf130
        del buf3
        buf140 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf141 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf157 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_15.run(buf10, buf140, buf141, buf157, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf137, (512, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf139, (512, 2048, 49), (100352, 49, 1), 0), buf140, buf141, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf144 = reinterpret_tensor(buf67, (512, 2048, 49), (100352, 49, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf137, (512, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf139, (512, 2048, 49), (100352, 49, 1), 0), rsqrt_46, primals_280, buf140, buf141, buf144, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del primals_280
        del rsqrt_46
        buf146 = empty_strided_cuda((401408, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_588, getitem_589, getitem_590, buf146, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_588
        del getitem_589
        del getitem_590
        buf148 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf149 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf144, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf148, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_47, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_47
        buf150 = buf149[0]
        assert_size_stride(buf150, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf150, 16, 'torch.ops.aten.convolution_backward.default')
        del buf149
        buf151 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf152 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf144, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf146, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf151, (2048, 1024, 1, 1), (0, 0, 0, 0), 0), None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf153 = buf152[1]
        assert_size_stride(buf153, (2048, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf153, 16, 'torch.ops.aten.convolution_backward.default')
        del buf152
        buf154 = empty_strided_cuda((2048, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_16.run(buf153, buf154, 2097152, stream=stream0)
        del buf153
        buf155 = reinterpret_tensor(buf144, (200704, 256), (256, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_585, getitem_586, getitem_587, buf155, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_585
        del getitem_586
        del getitem_587
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf155, (512, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf139, (512, 2048, 49), (100352, 49, 1), 0), buf10, buf157, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf160 = reinterpret_tensor(buf137, (512, 2048, 49), (100352, 49, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf155, (512, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf139, (512, 2048, 49), (100352, 49, 1), 0), rsqrt_45, primals_274, buf10, buf157, buf160, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del primals_274
        del rsqrt_45
        buf162 = reinterpret_tensor(buf123, (50176, 256), (256, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_578, getitem_579, getitem_580, buf162, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_578
        del getitem_579
        del getitem_580
        buf164 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf165 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf160, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf164, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_46
        buf166 = buf165[0]
        assert_size_stride(buf166, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf166, 16, 'torch.ops.aten.convolution_backward.default')
        del buf165
        buf167 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf168 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf160, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf162, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf167, (2048, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf169 = buf168[1]
        assert_size_stride(buf169, (2048, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf169, 16, 'torch.ops.aten.convolution_backward.default')
        del buf168
        buf170 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf169, buf170, 1048576, stream=stream0)
        del buf169
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_577, buf28, 8, 1, 256, 1, 1, 32, 8, 50176, 1, 1, stream=stream0)
        del buf114
        del buf29
        del buf51
        del buf93
        del getitem_577
        buf172 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_572, getitem_573, getitem_574, buf172, 16, 1, 256, 1, 2, 16, 16, 50176, 1, 1, stream=stream0)
        del getitem_572
        del getitem_573
        del getitem_574
        buf174 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf175 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf195 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf196 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(buf33, buf174, buf175, buf195, buf196, 512, stream=stream0)
        buf176 = buf124; del buf124  # reuse
        buf180 = reinterpret_tensor(buf116, (512, 512, 49), (25088, 49, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf166, buf28, buf176, buf180, 262144, 49, stream=stream0)
        del buf166
        del buf28
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf172, (512, 512, 49), (25088, 49, 1), 0), buf176, buf174, buf175, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf179 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf172, (512, 512, 49), (25088, 49, 1), 0), buf180, rsqrt_44, primals_268, buf174, buf175, buf179, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del buf172
        del buf180
        del primals_268
        del rsqrt_44
        buf182 = reinterpret_tensor(buf160, (200704, 256), (256, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_565, getitem_566, getitem_567, buf182, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_565
        del getitem_566
        del getitem_567
        buf184 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf185 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf179, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf184, (512, 512, 14, 14), (0, 0, 0, 0), 0), convert_element_type_45, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_45
        buf186 = buf185[0]
        assert_size_stride(buf186, (512, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf186, 16, 'torch.ops.aten.convolution_backward.default')
        del buf185
        buf187 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf188 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf179, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf182, (512, 512, 14, 14), (100352, 196, 14, 1), 0), reinterpret_tensor(buf187, (512, 512, 3, 3), (0, 0, 0, 0), 0), None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        del buf179
        buf189 = buf188[1]
        assert_size_stride(buf189, (512, 512, 3, 3), (4608, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf189, 16, 'torch.ops.aten.convolution_backward.default')
        del buf188
        buf190 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf189, buf190, 2359296, stream=stream0)
        del buf189
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_564, reinterpret_tensor(buf191, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_564
        buf193 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_559, getitem_560, getitem_561, buf193, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_559
        del getitem_560
        del getitem_561
        buf197 = reinterpret_tensor(buf155, (512, 512, 196), (100352, 196, 1), 0); del buf155  # reuse
        buf201 = reinterpret_tensor(buf139, (512, 512, 196), (100352, 196, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(buf186, buf191, buf197, buf201, 262144, 196, stream=stream0)
        del buf186
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_6.run(reinterpret_tensor(buf193, (512, 512, 196), (100352, 196, 1), 0), buf197, buf195, buf196, 100352, 196, 100352, 196, 1024, 512, 98, 1, stream=stream0)
        buf200 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_7.run(reinterpret_tensor(buf193, (512, 512, 196), (100352, 196, 1), 0), buf201, rsqrt_43, primals_262, buf195, buf196, buf200, 100352, 196, 100352, 196, 1024, 512, 98, 1, stream=stream0)
        del primals_262
        del rsqrt_43
        buf203 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_552, getitem_553, getitem_554, buf203, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_552
        del getitem_553
        del getitem_554
        buf205 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf206 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf200, (512, 512, 14, 14), (100352, 196, 14, 1), 0), reinterpret_tensor(buf205, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_44, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_44
        buf207 = buf206[0]
        assert_size_stride(buf207, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf207, 16, 'torch.ops.aten.convolution_backward.default')
        del buf206
        buf208 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf209 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf200, (512, 512, 14, 14), (100352, 196, 14, 1), 0), reinterpret_tensor(buf203, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf208, (512, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf210 = buf209[1]
        assert_size_stride(buf210, (512, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf210, 16, 'torch.ops.aten.convolution_backward.default')
        del buf209
        buf211 = empty_strided_cuda((512, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_18.run(buf210, buf211, 524288, stream=stream0)
        del buf210
        buf212 = empty_strided_cuda((401408, 256), (256, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_286], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_19.run(buf212, 102760448, stream=stream0)
        buf213 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_20.run(buf212, buf213, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_551, reinterpret_tensor(buf213, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_551
        buf215 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_546, getitem_547, getitem_548, buf215, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_546
        del getitem_547
        del getitem_548
        buf217 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_21.run(buf217, 1024, stream=stream0)
        buf218 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf219 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_22.run(buf217, buf218, buf219, 1024, stream=stream0)
        buf220 = empty_strided_cuda((512, 1024, 196), (200704, 196, 1), torch.bfloat16)
        buf224 = empty_strided_cuda((512, 1024, 196), (200704, 196, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_23.run(buf150, buf207, buf213, buf220, buf224, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf215, (512, 1024, 196), (200704, 196, 1), 0), buf220, buf218, buf219, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf223 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf215, (512, 1024, 196), (200704, 196, 1), 0), buf224, rsqrt_42, primals_256, buf218, buf219, buf223, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del buf215
        del buf224
        del primals_256
        del rsqrt_42
        buf226 = empty_strided_cuda((100352, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_539, getitem_540, getitem_541, buf226, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_539
        del getitem_540
        del getitem_541
        buf228 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf229 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf223, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf228, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_43, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_43
        buf230 = buf229[0]
        assert_size_stride(buf230, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf230, 16, 'torch.ops.aten.convolution_backward.default')
        del buf229
        buf231 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf232 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf223, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf226, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf231, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf233 = buf232[1]
        assert_size_stride(buf233, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf233, 16, 'torch.ops.aten.convolution_backward.default')
        del buf232
        buf234 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf233, buf234, 262144, stream=stream0)
        del buf233
        buf235 = empty_strided_cuda((100352, 256), (256, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_289], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_25.run(buf235, 25690112, stream=stream0)
        buf236 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf258 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf235, buf236, buf258, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_538, reinterpret_tensor(buf236, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_538
        buf238 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_533, getitem_534, getitem_535, buf238, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_533
        del getitem_534
        del getitem_535
        buf240 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_27.run(buf240, 256, stream=stream0)
        buf241 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf242 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf262 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf263 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf240, buf241, buf242, buf262, buf263, 256, stream=stream0)
        buf243 = empty_strided_cuda((512, 256, 196), (50176, 196, 1), torch.bfloat16)
        buf247 = empty_strided_cuda((512, 256, 196), (50176, 196, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf230, buf236, buf243, buf247, 131072, 196, stream=stream0)
        del buf230
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf238, (512, 256, 196), (50176, 196, 1), 0), buf243, buf241, buf242, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf246 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf238, (512, 256, 196), (50176, 196, 1), 0), buf247, rsqrt_41, primals_250, buf241, buf242, buf246, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_250
        del rsqrt_41
        buf249 = reinterpret_tensor(buf247, (100352, 256), (256, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_526, getitem_527, getitem_528, buf249, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_526
        del getitem_527
        del getitem_528
        buf251 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf252 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf246, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf251, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_42, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_42
        buf253 = buf252[0]
        assert_size_stride(buf253, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf253, 16, 'torch.ops.aten.convolution_backward.default')
        del buf252
        buf254 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf255 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf246, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf249, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf254, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf256 = buf255[1]
        assert_size_stride(buf256, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf256, 16, 'torch.ops.aten.convolution_backward.default')
        del buf255
        buf257 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf256, buf257, 589824, stream=stream0)
        del buf256
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_525, reinterpret_tensor(buf258, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_525
        buf260 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_520, getitem_521, getitem_522, buf260, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_520
        del getitem_521
        del getitem_522
        buf264 = buf246; del buf246  # reuse
        buf268 = reinterpret_tensor(buf238, (512, 256, 196), (50176, 196, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf253, buf258, buf264, buf268, 131072, 196, stream=stream0)
        del buf253
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf260, (512, 256, 196), (50176, 196, 1), 0), buf264, buf262, buf263, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf267 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf260, (512, 256, 196), (50176, 196, 1), 0), buf268, rsqrt_40, primals_244, buf262, buf263, buf267, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_244
        del rsqrt_40
        buf270 = reinterpret_tensor(buf223, (401408, 256), (256, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_513, getitem_514, getitem_515, buf270, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_513
        del getitem_514
        del getitem_515
        buf272 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf273 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf267, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf272, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_41
        buf274 = buf273[0]
        assert_size_stride(buf274, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf274, 16, 'torch.ops.aten.convolution_backward.default')
        del buf273
        buf275 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf276 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf267, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf270, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf275, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf277 = buf276[1]
        assert_size_stride(buf277, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf277, 16, 'torch.ops.aten.convolution_backward.default')
        del buf276
        buf278 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf277, buf278, 262144, stream=stream0)
        del buf277
        buf279 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf341 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_31.run(buf212, buf279, buf341, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_512, reinterpret_tensor(buf279, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_512
        buf281 = reinterpret_tensor(buf270, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf270  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_32.run(buf150, buf207, buf213, buf274, buf279, buf281, 524288, 196, stream=stream0)
        buf282 = reinterpret_tensor(buf274, (401408, 256), (256, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_507, getitem_508, getitem_509, buf282, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_507
        del getitem_508
        del getitem_509
        buf284 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf285 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf345 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf346 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_33.run(buf217, buf284, buf285, buf345, buf346, 1024, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf282, (512, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf281, (512, 1024, 196), (200704, 196, 1), 0), buf284, buf285, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf288 = reinterpret_tensor(buf207, (512, 1024, 196), (200704, 196, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf282, (512, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf281, (512, 1024, 196), (200704, 196, 1), 0), rsqrt_39, primals_238, buf284, buf285, buf288, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_238
        del rsqrt_39
        buf290 = reinterpret_tensor(buf267, (100352, 256), (256, 1), 0); del buf267  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_500, getitem_501, getitem_502, buf290, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_500
        del getitem_501
        del getitem_502
        buf292 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf293 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf288, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf292, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_40
        buf294 = buf293[0]
        assert_size_stride(buf294, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf294, 16, 'torch.ops.aten.convolution_backward.default')
        del buf293
        buf295 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf296 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf288, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf290, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf295, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf297 = buf296[1]
        assert_size_stride(buf297, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf297, 16, 'torch.ops.aten.convolution_backward.default')
        del buf296
        buf298 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf297, buf298, 262144, stream=stream0)
        del buf297
        buf299 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf320 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf235, buf299, buf320, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_499, reinterpret_tensor(buf299, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_499
        buf301 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_494, getitem_495, getitem_496, buf301, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_494
        del getitem_495
        del getitem_496
        buf303 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf304 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf324 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf325 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf240, buf303, buf304, buf324, buf325, 256, stream=stream0)
        buf305 = buf268; del buf268  # reuse
        buf309 = reinterpret_tensor(buf260, (512, 256, 196), (50176, 196, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf294, buf299, buf305, buf309, 131072, 196, stream=stream0)
        del buf294
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf301, (512, 256, 196), (50176, 196, 1), 0), buf305, buf303, buf304, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf308 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf301, (512, 256, 196), (50176, 196, 1), 0), buf309, rsqrt_38, primals_232, buf303, buf304, buf308, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_232
        del rsqrt_38
        buf311 = reinterpret_tensor(buf309, (100352, 256), (256, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_487, getitem_488, getitem_489, buf311, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_487
        del getitem_488
        del getitem_489
        buf313 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf314 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf308, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf313, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_39
        buf315 = buf314[0]
        assert_size_stride(buf315, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf315, 16, 'torch.ops.aten.convolution_backward.default')
        del buf314
        buf316 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf317 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf308, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf311, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf316, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf318 = buf317[1]
        assert_size_stride(buf318, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf318, 16, 'torch.ops.aten.convolution_backward.default')
        del buf317
        buf319 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf318, buf319, 589824, stream=stream0)
        del buf318
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_486, reinterpret_tensor(buf320, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_486
        buf322 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_481, getitem_482, getitem_483, buf322, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_481
        del getitem_482
        del getitem_483
        buf326 = buf308; del buf308  # reuse
        buf330 = reinterpret_tensor(buf301, (512, 256, 196), (50176, 196, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf315, buf320, buf326, buf330, 131072, 196, stream=stream0)
        del buf315
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf322, (512, 256, 196), (50176, 196, 1), 0), buf326, buf324, buf325, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf329 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf322, (512, 256, 196), (50176, 196, 1), 0), buf330, rsqrt_37, primals_226, buf324, buf325, buf329, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_226
        del rsqrt_37
        buf332 = reinterpret_tensor(buf288, (401408, 256), (256, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_474, getitem_475, getitem_476, buf332, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_474
        del getitem_475
        del getitem_476
        buf334 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf335 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf329, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf334, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_38, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_38
        buf336 = buf335[0]
        assert_size_stride(buf336, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf336, 16, 'torch.ops.aten.convolution_backward.default')
        del buf335
        buf337 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf338 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf329, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf332, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf337, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf339 = buf338[1]
        assert_size_stride(buf339, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf339, 16, 'torch.ops.aten.convolution_backward.default')
        del buf338
        buf340 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf339, buf340, 262144, stream=stream0)
        del buf339
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_473, reinterpret_tensor(buf341, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_473
        buf343 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_468, getitem_469, getitem_470, buf343, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_468
        del getitem_469
        del getitem_470
        buf347 = reinterpret_tensor(buf282, (512, 1024, 196), (200704, 196, 1), 0); del buf282  # reuse
        buf351 = reinterpret_tensor(buf150, (512, 1024, 196), (200704, 196, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf281, buf336, buf341, buf347, buf351, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf343, (512, 1024, 196), (200704, 196, 1), 0), buf347, buf345, buf346, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf350 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf343, (512, 1024, 196), (200704, 196, 1), 0), buf351, rsqrt_36, primals_220, buf345, buf346, buf350, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_220
        del rsqrt_36
        buf353 = reinterpret_tensor(buf329, (100352, 256), (256, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_461, getitem_462, getitem_463, buf353, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_461
        del getitem_462
        del getitem_463
        buf355 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf356 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf350, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf355, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_37
        buf357 = buf356[0]
        assert_size_stride(buf357, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf357, 16, 'torch.ops.aten.convolution_backward.default')
        del buf356
        buf358 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf359 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf350, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf353, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf358, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf360 = buf359[1]
        assert_size_stride(buf360, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf360, 16, 'torch.ops.aten.convolution_backward.default')
        del buf359
        buf361 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf360, buf361, 262144, stream=stream0)
        del buf360
        buf362 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf383 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf235, buf362, buf383, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_460, reinterpret_tensor(buf362, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_460
        buf364 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_455, getitem_456, getitem_457, buf364, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_455
        del getitem_456
        del getitem_457
        buf366 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf367 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf387 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf388 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf240, buf366, buf367, buf387, buf388, 256, stream=stream0)
        buf368 = buf330; del buf330  # reuse
        buf372 = reinterpret_tensor(buf322, (512, 256, 196), (50176, 196, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf357, buf362, buf368, buf372, 131072, 196, stream=stream0)
        del buf357
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf364, (512, 256, 196), (50176, 196, 1), 0), buf368, buf366, buf367, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf371 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf364, (512, 256, 196), (50176, 196, 1), 0), buf372, rsqrt_35, primals_214, buf366, buf367, buf371, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_214
        del rsqrt_35
        buf374 = reinterpret_tensor(buf372, (100352, 256), (256, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_448, getitem_449, getitem_450, buf374, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_448
        del getitem_449
        del getitem_450
        buf376 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf377 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf371, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf376, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_36, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_36
        buf378 = buf377[0]
        assert_size_stride(buf378, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf378, 16, 'torch.ops.aten.convolution_backward.default')
        del buf377
        buf379 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf380 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf371, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf374, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf379, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf381 = buf380[1]
        assert_size_stride(buf381, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf381, 16, 'torch.ops.aten.convolution_backward.default')
        del buf380
        buf382 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf381, buf382, 589824, stream=stream0)
        del buf381
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_447, reinterpret_tensor(buf383, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_447
        buf385 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_442, getitem_443, getitem_444, buf385, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_442
        del getitem_443
        del getitem_444
        buf389 = buf371; del buf371  # reuse
        buf393 = reinterpret_tensor(buf364, (512, 256, 196), (50176, 196, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf378, buf383, buf389, buf393, 131072, 196, stream=stream0)
        del buf378
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf385, (512, 256, 196), (50176, 196, 1), 0), buf389, buf387, buf388, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf392 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf385, (512, 256, 196), (50176, 196, 1), 0), buf393, rsqrt_34, primals_208, buf387, buf388, buf392, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_208
        del rsqrt_34
        buf395 = reinterpret_tensor(buf350, (401408, 256), (256, 1), 0); del buf350  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_435, getitem_436, getitem_437, buf395, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_435
        del getitem_436
        del getitem_437
        buf397 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf398 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf392, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf397, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_35, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_35
        buf399 = buf398[0]
        assert_size_stride(buf399, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf399, 16, 'torch.ops.aten.convolution_backward.default')
        del buf398
        buf400 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf401 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf392, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf395, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf400, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf402 = buf401[1]
        assert_size_stride(buf402, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf402, 16, 'torch.ops.aten.convolution_backward.default')
        del buf401
        buf403 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf402, buf403, 262144, stream=stream0)
        del buf402
        buf404 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf466 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_31.run(buf212, buf404, buf466, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_434, reinterpret_tensor(buf404, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_434
        buf406 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_35.run(buf406, buf336, buf341, buf399, buf404, 524288, 196, stream=stream0)
        buf407 = reinterpret_tensor(buf399, (401408, 256), (256, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_429, getitem_430, getitem_431, buf407, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_429
        del getitem_430
        del getitem_431
        buf409 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf410 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf470 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf471 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_33.run(buf217, buf409, buf410, buf470, buf471, 1024, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf407, (512, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf406, (512, 1024, 196), (200704, 196, 1), 0), buf409, buf410, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf413 = reinterpret_tensor(buf336, (512, 1024, 196), (200704, 196, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf407, (512, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf406, (512, 1024, 196), (200704, 196, 1), 0), rsqrt_33, primals_202, buf409, buf410, buf413, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_202
        del rsqrt_33
        buf415 = reinterpret_tensor(buf392, (100352, 256), (256, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_422, getitem_423, getitem_424, buf415, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_422
        del getitem_423
        del getitem_424
        buf417 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf418 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf413, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf417, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_34
        buf419 = buf418[0]
        assert_size_stride(buf419, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf419, 16, 'torch.ops.aten.convolution_backward.default')
        del buf418
        buf420 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf421 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf413, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf415, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf420, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf422 = buf421[1]
        assert_size_stride(buf422, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf422, 16, 'torch.ops.aten.convolution_backward.default')
        del buf421
        buf423 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf422, buf423, 262144, stream=stream0)
        del buf422
        buf424 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf445 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf235, buf424, buf445, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_421, reinterpret_tensor(buf424, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_421
        buf426 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_416, getitem_417, getitem_418, buf426, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_416
        del getitem_417
        del getitem_418
        buf428 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf429 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf449 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf450 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf240, buf428, buf429, buf449, buf450, 256, stream=stream0)
        buf430 = buf393; del buf393  # reuse
        buf434 = reinterpret_tensor(buf385, (512, 256, 196), (50176, 196, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf419, buf424, buf430, buf434, 131072, 196, stream=stream0)
        del buf419
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf426, (512, 256, 196), (50176, 196, 1), 0), buf430, buf428, buf429, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf433 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf426, (512, 256, 196), (50176, 196, 1), 0), buf434, rsqrt_32, primals_196, buf428, buf429, buf433, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_196
        del rsqrt_32
        buf436 = reinterpret_tensor(buf434, (100352, 256), (256, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_409, getitem_410, getitem_411, buf436, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_409
        del getitem_410
        del getitem_411
        buf438 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf439 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf433, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf438, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_33
        buf440 = buf439[0]
        assert_size_stride(buf440, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf440, 16, 'torch.ops.aten.convolution_backward.default')
        del buf439
        buf441 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf442 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf433, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf436, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf441, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf443 = buf442[1]
        assert_size_stride(buf443, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf443, 16, 'torch.ops.aten.convolution_backward.default')
        del buf442
        buf444 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf443, buf444, 589824, stream=stream0)
        del buf443
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_408, reinterpret_tensor(buf445, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_408
        buf447 = buf436; del buf436  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_403, getitem_404, getitem_405, buf447, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_403
        del getitem_404
        del getitem_405
        buf451 = buf433; del buf433  # reuse
        buf455 = reinterpret_tensor(buf426, (512, 256, 196), (50176, 196, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf440, buf445, buf451, buf455, 131072, 196, stream=stream0)
        del buf440
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf447, (512, 256, 196), (50176, 196, 1), 0), buf451, buf449, buf450, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf454 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf447, (512, 256, 196), (50176, 196, 1), 0), buf455, rsqrt_31, primals_190, buf449, buf450, buf454, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_190
        del rsqrt_31
        buf457 = reinterpret_tensor(buf413, (401408, 256), (256, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_396, getitem_397, getitem_398, buf457, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_396
        del getitem_397
        del getitem_398
        buf459 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf460 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf454, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf459, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_32, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_32
        buf461 = buf460[0]
        assert_size_stride(buf461, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf461, 16, 'torch.ops.aten.convolution_backward.default')
        del buf460
        buf462 = buf459; del buf459  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf463 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf454, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf457, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf462, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf464 = buf463[1]
        assert_size_stride(buf464, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf464, 16, 'torch.ops.aten.convolution_backward.default')
        del buf463
        buf465 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf464, buf465, 262144, stream=stream0)
        del buf464
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_395, reinterpret_tensor(buf466, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_395
        buf468 = buf457; del buf457  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_390, getitem_391, getitem_392, buf468, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_390
        del getitem_391
        del getitem_392
        buf472 = reinterpret_tensor(buf407, (512, 1024, 196), (200704, 196, 1), 0); del buf407  # reuse
        buf476 = reinterpret_tensor(buf395, (512, 1024, 196), (200704, 196, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf406, buf461, buf466, buf472, buf476, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf468, (512, 1024, 196), (200704, 196, 1), 0), buf472, buf470, buf471, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf475 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf468, (512, 1024, 196), (200704, 196, 1), 0), buf476, rsqrt_30, primals_184, buf470, buf471, buf475, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_184
        del rsqrt_30
        buf478 = reinterpret_tensor(buf454, (100352, 256), (256, 1), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_383, getitem_384, getitem_385, buf478, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_383
        del getitem_384
        del getitem_385
        buf480 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf481 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf475, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf480, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_31
        buf482 = buf481[0]
        assert_size_stride(buf482, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf482, 16, 'torch.ops.aten.convolution_backward.default')
        del buf481
        buf483 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf484 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf475, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf478, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf483, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf485 = buf484[1]
        assert_size_stride(buf485, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf485, 16, 'torch.ops.aten.convolution_backward.default')
        del buf484
        buf486 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf485, buf486, 262144, stream=stream0)
        del buf485
        buf487 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf508 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_26.run(buf235, buf487, buf508, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_382, reinterpret_tensor(buf487, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_382
        buf489 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_377, getitem_378, getitem_379, buf489, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_377
        del getitem_378
        del getitem_379
        buf491 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf492 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf512 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf513 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf240, buf491, buf492, buf512, buf513, 256, stream=stream0)
        buf493 = buf455; del buf455  # reuse
        buf497 = reinterpret_tensor(buf447, (512, 256, 196), (50176, 196, 1), 0); del buf447  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf482, buf487, buf493, buf497, 131072, 196, stream=stream0)
        del buf482
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf489, (512, 256, 196), (50176, 196, 1), 0), buf493, buf491, buf492, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf496 = buf493; del buf493  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf489, (512, 256, 196), (50176, 196, 1), 0), buf497, rsqrt_29, primals_178, buf491, buf492, buf496, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_178
        del rsqrt_29
        buf499 = reinterpret_tensor(buf497, (100352, 256), (256, 1), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_370, getitem_371, getitem_372, buf499, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_370
        del getitem_371
        del getitem_372
        buf501 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf502 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf496, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf501, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_30, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_30
        buf503 = buf502[0]
        assert_size_stride(buf503, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf503, 16, 'torch.ops.aten.convolution_backward.default')
        del buf502
        buf504 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf505 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf496, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf499, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf504, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf506 = buf505[1]
        assert_size_stride(buf506, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf506, 16, 'torch.ops.aten.convolution_backward.default')
        del buf505
        buf507 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf506, buf507, 589824, stream=stream0)
        del buf506
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_369, reinterpret_tensor(buf508, (100352, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del getitem_369
        buf510 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_364, getitem_365, getitem_366, buf510, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_364
        del getitem_365
        del getitem_366
        buf514 = buf496; del buf496  # reuse
        buf518 = reinterpret_tensor(buf489, (512, 256, 196), (50176, 196, 1), 0); del buf489  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf503, buf508, buf514, buf518, 131072, 196, stream=stream0)
        del buf503
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf510, (512, 256, 196), (50176, 196, 1), 0), buf514, buf512, buf513, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf517 = buf514; del buf514  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf510, (512, 256, 196), (50176, 196, 1), 0), buf518, rsqrt_28, primals_172, buf512, buf513, buf517, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_172
        del rsqrt_28
        buf520 = reinterpret_tensor(buf475, (401408, 256), (256, 1), 0); del buf475  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_357, getitem_358, getitem_359, buf520, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_357
        del getitem_358
        del getitem_359
        buf522 = buf504; del buf504  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf523 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf517, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf522, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_29
        buf524 = buf523[0]
        assert_size_stride(buf524, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf524, 16, 'torch.ops.aten.convolution_backward.default')
        del buf523
        buf525 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf526 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf517, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf520, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf525, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf527 = buf526[1]
        assert_size_stride(buf527, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf527, 16, 'torch.ops.aten.convolution_backward.default')
        del buf526
        buf528 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf527, buf528, 262144, stream=stream0)
        del buf527
        buf529 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf589 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_31.run(buf212, buf529, buf589, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_356, reinterpret_tensor(buf529, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_356
        buf531 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_351, getitem_352, getitem_353, buf531, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_351
        del getitem_352
        del getitem_353
        buf533 = buf461; del buf461  # reuse
        buf536 = buf476; del buf476  # reuse
        buf540 = reinterpret_tensor(buf468, (512, 1024, 196), (200704, 196, 1), 0); del buf468  # reuse
        buf554 = buf351; del buf351  # reuse
        buf558 = reinterpret_tensor(buf343, (512, 1024, 196), (200704, 196, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_36.run(buf533, buf406, buf466, buf524, buf529, buf536, buf540, buf554, buf558, 524288, 196, stream=stream0)
        del buf406
        del buf524
        del buf533
        buf534 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf535 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf553 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_37.run(buf217, buf534, buf535, buf553, 1024, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf531, (512, 1024, 196), (200704, 196, 1), 0), buf536, buf534, buf535, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf539 = buf536; del buf536  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf531, (512, 1024, 196), (200704, 196, 1), 0), buf540, rsqrt_27, primals_166, buf534, buf535, buf539, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del buf531
        del buf540
        del primals_166
        del rsqrt_27
        buf542 = empty_strided_cuda((802816, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_344, getitem_345, getitem_346, buf542, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_344
        del getitem_345
        del getitem_346
        buf544 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf545 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf539, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf544, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_28, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_28
        buf546 = buf545[0]
        assert_size_stride(buf546, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf546, 16, 'torch.ops.aten.convolution_backward.default')
        del buf545
        buf547 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf548 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf539, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf542, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf547, (1024, 512, 1, 1), (0, 0, 0, 0), 0), None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf549 = buf548[1]
        assert_size_stride(buf549, (1024, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf549, 16, 'torch.ops.aten.convolution_backward.default')
        del buf548
        buf550 = empty_strided_cuda((1024, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_18.run(buf549, buf550, 524288, stream=stream0)
        del buf549
        buf551 = reinterpret_tensor(buf539, (401408, 256), (256, 1), 0); del buf539  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_341, getitem_342, getitem_343, buf551, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_341
        del getitem_342
        del getitem_343
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf551, (512, 1024, 196), (200704, 196, 1), 0), buf554, buf217, buf553, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf557 = buf554; del buf554  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf551, (512, 1024, 196), (200704, 196, 1), 0), buf558, rsqrt_26, primals_160, buf217, buf553, buf557, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_160
        del rsqrt_26
        buf560 = reinterpret_tensor(buf517, (100352, 256), (256, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_334, getitem_335, getitem_336, buf560, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_334
        del getitem_335
        del getitem_336
        buf562 = buf547; del buf547  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf563 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf557, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf562, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_27
        buf564 = buf563[0]
        assert_size_stride(buf564, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf564, 16, 'torch.ops.aten.convolution_backward.default')
        del buf563
        buf565 = buf562; del buf562  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf566 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf557, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf560, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf565, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf567 = buf566[1]
        assert_size_stride(buf567, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf567, 16, 'torch.ops.aten.convolution_backward.default')
        del buf566
        buf568 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf567, buf568, 262144, stream=stream0)
        del buf567
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_333, buf235, 8, 1, 256, 1, 1, 32, 8, 100352, 1, 1, stream=stream0)
        del buf236
        del buf258
        del buf299
        del buf320
        del buf362
        del buf383
        del buf424
        del buf445
        del buf487
        del buf508
        del getitem_333
        buf570 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_328, getitem_329, getitem_330, buf570, 16, 1, 256, 1, 2, 16, 16, 100352, 1, 1, stream=stream0)
        del getitem_328
        del getitem_329
        del getitem_330
        buf572 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf573 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf593 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf594 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf240, buf572, buf573, buf593, buf594, 256, stream=stream0)
        buf574 = buf518; del buf518  # reuse
        buf578 = reinterpret_tensor(buf510, (512, 256, 196), (50176, 196, 1), 0); del buf510  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf564, buf235, buf574, buf578, 131072, 196, stream=stream0)
        del buf235
        del buf564
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf570, (512, 256, 196), (50176, 196, 1), 0), buf574, buf572, buf573, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf577 = buf574; del buf574  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf570, (512, 256, 196), (50176, 196, 1), 0), buf578, rsqrt_25, primals_154, buf572, buf573, buf577, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf570
        del buf578
        del primals_154
        del rsqrt_25
        buf580 = reinterpret_tensor(buf557, (401408, 256), (256, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_321, getitem_322, getitem_323, buf580, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_321
        del getitem_322
        del getitem_323
        buf582 = buf565; del buf565  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf583 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf577, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf582, (512, 256, 28, 28), (0, 0, 0, 0), 0), convert_element_type_26, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_26
        buf584 = buf583[0]
        assert_size_stride(buf584, (512, 256, 28, 28), (200704, 1, 7168, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf584, 16, 'torch.ops.aten.convolution_backward.default')
        del buf583
        buf585 = buf582; del buf582  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf586 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf577, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf580, (512, 256, 28, 28), (200704, 784, 28, 1), 0), reinterpret_tensor(buf585, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        del buf577
        buf587 = buf586[1]
        assert_size_stride(buf587, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf587, 16, 'torch.ops.aten.convolution_backward.default')
        del buf586
        buf588 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_30.run(buf587, buf588, 589824, stream=stream0)
        del buf587
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_320, reinterpret_tensor(buf589, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_320
        buf591 = buf580; del buf580  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_315, getitem_316, getitem_317, buf591, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_315
        del getitem_316
        del getitem_317
        buf595 = reinterpret_tensor(buf558, (512, 256, 784), (200704, 784, 1), 0); del buf558  # reuse
        buf599 = reinterpret_tensor(buf551, (512, 256, 784), (200704, 784, 1), 0); del buf551  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf584, buf589, buf595, buf599, 131072, 784, stream=stream0)
        del buf584
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_12.run(reinterpret_tensor(buf591, (512, 256, 784), (200704, 784, 1), 0), buf595, buf593, buf594, 401408, 784, 200704, 784, 1024, 256, 392, 1, stream=stream0)
        buf598 = buf595; del buf595  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_13.run(reinterpret_tensor(buf591, (512, 256, 784), (200704, 784, 1), 0), buf599, rsqrt_24, primals_148, buf593, buf594, buf598, 401408, 784, 200704, 784, 1024, 256, 392, 1, stream=stream0)
        del primals_148
        del rsqrt_24
        buf601 = buf542; del buf542  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_308, getitem_309, getitem_310, buf601, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_308
        del getitem_309
        del getitem_310
        buf603 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf604 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf598, (512, 256, 28, 28), (200704, 784, 28, 1), 0), reinterpret_tensor(buf603, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_25
        buf605 = buf604[0]
        assert_size_stride(buf605, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf605, 16, 'torch.ops.aten.convolution_backward.default')
        del buf604
        buf606 = buf603; del buf603  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf607 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf598, (512, 256, 28, 28), (200704, 784, 28, 1), 0), reinterpret_tensor(buf601, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf606, (256, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf608 = buf607[1]
        assert_size_stride(buf608, (256, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf608, 16, 'torch.ops.aten.convolution_backward.default')
        del buf607
        buf609 = empty_strided_cuda((256, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_39.run(buf608, buf609, 131072, stream=stream0)
        del buf608
        buf610 = empty_strided_cuda((802816, 256), (256, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_342], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_40.run(buf610, 205520896, stream=stream0)
        buf611 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_41.run(buf610, buf611, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_307, reinterpret_tensor(buf611, (802816, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 802816, 1, 1, stream=stream0)
        del getitem_307
        buf613 = buf601; del buf601  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_302, getitem_303, getitem_304, buf613, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_302
        del getitem_303
        del getitem_304
        buf615 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf616 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_42.run(buf33, buf615, buf616, 512, stream=stream0)
        buf617 = empty_strided_cuda((512, 512, 784), (401408, 784, 1), torch.bfloat16)
        buf621 = empty_strided_cuda((512, 512, 784), (401408, 784, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf546, buf605, buf611, buf617, buf621, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf613, (512, 512, 784), (401408, 784, 1), 0), buf617, buf615, buf616, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf620 = buf617; del buf617  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf613, (512, 512, 784), (401408, 784, 1), 0), buf621, rsqrt_23, primals_142, buf615, buf616, buf620, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_142
        del rsqrt_23
        buf623 = reinterpret_tensor(buf200, (200704, 256), (256, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_295, getitem_296, getitem_297, buf623, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_295
        del getitem_296
        del getitem_297
        buf625 = buf606; del buf606  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf626 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf620, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf625, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_24
        buf627 = buf626[0]
        assert_size_stride(buf627, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf627, 16, 'torch.ops.aten.convolution_backward.default')
        del buf626
        buf628 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf629 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf620, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf623, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf628, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf630 = buf629[1]
        assert_size_stride(buf630, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf630, 16, 'torch.ops.aten.convolution_backward.default')
        del buf629
        buf631 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(buf630, buf631, 65536, stream=stream0)
        del buf630
        buf632 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf654 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf5, buf632, buf654, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_294, reinterpret_tensor(buf632, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_294
        buf634 = buf623; del buf623  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_289, getitem_290, getitem_291, buf634, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_289
        del getitem_290
        del getitem_291
        buf636 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_45.run(buf636, 128, stream=stream0)
        buf637 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf638 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf658 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf659 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_46.run(buf636, buf637, buf638, buf658, buf659, 128, stream=stream0)
        buf639 = reinterpret_tensor(buf201, (512, 128, 784), (100352, 784, 1), 0); del buf201  # reuse
        buf643 = reinterpret_tensor(buf193, (512, 128, 784), (100352, 784, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf627, buf632, buf639, buf643, 65536, 784, stream=stream0)
        del buf627
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf634, (512, 128, 784), (100352, 784, 1), 0), buf639, buf637, buf638, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf642 = buf639; del buf639  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf634, (512, 128, 784), (100352, 784, 1), 0), buf643, rsqrt_22, primals_136, buf637, buf638, buf642, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_136
        del rsqrt_22
        buf645 = reinterpret_tensor(buf643, (200704, 256), (256, 1), 0); del buf643  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_282, getitem_283, getitem_284, buf645, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_282
        del getitem_283
        del getitem_284
        buf647 = buf628; del buf628  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf648 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf642, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf647, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_23, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_23
        buf649 = buf648[0]
        assert_size_stride(buf649, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf649, 16, 'torch.ops.aten.convolution_backward.default')
        del buf648
        buf650 = buf647; del buf647  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf651 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf642, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf645, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf650, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf652 = buf651[1]
        assert_size_stride(buf652, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf652, 16, 'torch.ops.aten.convolution_backward.default')
        del buf651
        buf653 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_48.run(buf652, buf653, 147456, stream=stream0)
        del buf652
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_281, reinterpret_tensor(buf654, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_281
        buf656 = buf645; del buf645  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_276, getitem_277, getitem_278, buf656, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_276
        del getitem_277
        del getitem_278
        buf660 = buf642; del buf642  # reuse
        buf664 = reinterpret_tensor(buf634, (512, 128, 784), (100352, 784, 1), 0); del buf634  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf649, buf654, buf660, buf664, 65536, 784, stream=stream0)
        del buf649
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf656, (512, 128, 784), (100352, 784, 1), 0), buf660, buf658, buf659, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf663 = buf660; del buf660  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf656, (512, 128, 784), (100352, 784, 1), 0), buf664, rsqrt_21, primals_130, buf658, buf659, buf663, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_130
        del rsqrt_21
        buf666 = reinterpret_tensor(buf620, (802816, 256), (256, 1), 0); del buf620  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_269, getitem_270, getitem_271, buf666, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_269
        del getitem_270
        del getitem_271
        buf668 = buf650; del buf650  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf669 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf663, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf668, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_22, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_22
        buf670 = buf669[0]
        assert_size_stride(buf670, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf670, 16, 'torch.ops.aten.convolution_backward.default')
        del buf669
        buf671 = buf668; del buf668  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf672 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf663, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf666, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf671, (128, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf673 = buf672[1]
        assert_size_stride(buf673, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf673, 16, 'torch.ops.aten.convolution_backward.default')
        del buf672
        buf674 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(buf673, buf674, 65536, stream=stream0)
        del buf673
        buf675 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        buf737 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_49.run(buf610, buf675, buf737, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_268, reinterpret_tensor(buf675, (802816, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 802816, 1, 1, stream=stream0)
        del getitem_268
        buf677 = reinterpret_tensor(buf666, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf666  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_50.run(buf546, buf605, buf611, buf670, buf675, buf677, 262144, 784, stream=stream0)
        buf678 = reinterpret_tensor(buf670, (802816, 256), (256, 1), 0); del buf670  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_263, getitem_264, getitem_265, buf678, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_263
        del getitem_264
        del getitem_265
        buf680 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf681 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf741 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf742 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_10.run(buf33, buf680, buf681, buf741, buf742, 512, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf678, (512, 512, 784), (401408, 784, 1), 0), reinterpret_tensor(buf677, (512, 512, 784), (401408, 784, 1), 0), buf680, buf681, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf684 = reinterpret_tensor(buf605, (512, 512, 784), (401408, 784, 1), 0); del buf605  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf678, (512, 512, 784), (401408, 784, 1), 0), reinterpret_tensor(buf677, (512, 512, 784), (401408, 784, 1), 0), rsqrt_20, primals_124, buf680, buf681, buf684, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_124
        del rsqrt_20
        buf686 = reinterpret_tensor(buf663, (200704, 256), (256, 1), 0); del buf663  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_256, getitem_257, getitem_258, buf686, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_256
        del getitem_257
        del getitem_258
        buf688 = buf671; del buf671  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf689 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf684, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf688, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_21
        buf690 = buf689[0]
        assert_size_stride(buf690, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf690, 16, 'torch.ops.aten.convolution_backward.default')
        del buf689
        buf691 = buf688; del buf688  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf692 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf684, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf686, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf691, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf693 = buf692[1]
        assert_size_stride(buf693, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf693, 16, 'torch.ops.aten.convolution_backward.default')
        del buf692
        buf694 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(buf693, buf694, 65536, stream=stream0)
        del buf693
        buf695 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf716 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf5, buf695, buf716, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_255, reinterpret_tensor(buf695, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_255
        buf697 = buf686; del buf686  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_250, getitem_251, getitem_252, buf697, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_250
        del getitem_251
        del getitem_252
        buf699 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf700 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf720 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf721 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_46.run(buf636, buf699, buf700, buf720, buf721, 128, stream=stream0)
        buf701 = buf664; del buf664  # reuse
        buf705 = reinterpret_tensor(buf656, (512, 128, 784), (100352, 784, 1), 0); del buf656  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf690, buf695, buf701, buf705, 65536, 784, stream=stream0)
        del buf690
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf697, (512, 128, 784), (100352, 784, 1), 0), buf701, buf699, buf700, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf704 = buf701; del buf701  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf697, (512, 128, 784), (100352, 784, 1), 0), buf705, rsqrt_19, primals_118, buf699, buf700, buf704, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_118
        del rsqrt_19
        buf707 = reinterpret_tensor(buf705, (200704, 256), (256, 1), 0); del buf705  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_243, getitem_244, getitem_245, buf707, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_243
        del getitem_244
        del getitem_245
        buf709 = buf691; del buf691  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf710 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf704, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf709, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_20, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_20
        buf711 = buf710[0]
        assert_size_stride(buf711, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf711, 16, 'torch.ops.aten.convolution_backward.default')
        del buf710
        buf712 = buf709; del buf709  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf713 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf704, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf707, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf712, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf714 = buf713[1]
        assert_size_stride(buf714, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf714, 16, 'torch.ops.aten.convolution_backward.default')
        del buf713
        buf715 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_48.run(buf714, buf715, 147456, stream=stream0)
        del buf714
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_242, reinterpret_tensor(buf716, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_242
        buf718 = buf707; del buf707  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_237, getitem_238, getitem_239, buf718, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_237
        del getitem_238
        del getitem_239
        buf722 = buf704; del buf704  # reuse
        buf726 = reinterpret_tensor(buf697, (512, 128, 784), (100352, 784, 1), 0); del buf697  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf711, buf716, buf722, buf726, 65536, 784, stream=stream0)
        del buf711
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf718, (512, 128, 784), (100352, 784, 1), 0), buf722, buf720, buf721, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf725 = buf722; del buf722  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf718, (512, 128, 784), (100352, 784, 1), 0), buf726, rsqrt_18, primals_112, buf720, buf721, buf725, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_112
        del rsqrt_18
        buf728 = reinterpret_tensor(buf684, (802816, 256), (256, 1), 0); del buf684  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_230, getitem_231, getitem_232, buf728, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_230
        del getitem_231
        del getitem_232
        buf730 = buf712; del buf712  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf731 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf725, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf730, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_19
        buf732 = buf731[0]
        assert_size_stride(buf732, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf732, 16, 'torch.ops.aten.convolution_backward.default')
        del buf731
        buf733 = buf730; del buf730  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf734 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf725, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf728, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf733, (128, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf735 = buf734[1]
        assert_size_stride(buf735, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf735, 16, 'torch.ops.aten.convolution_backward.default')
        del buf734
        buf736 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(buf735, buf736, 65536, stream=stream0)
        del buf735
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_229, reinterpret_tensor(buf737, (802816, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 802816, 1, 1, stream=stream0)
        del getitem_229
        buf739 = buf728; del buf728  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_224, getitem_225, getitem_226, buf739, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_224
        del getitem_225
        del getitem_226
        buf743 = reinterpret_tensor(buf678, (512, 512, 784), (401408, 784, 1), 0); del buf678  # reuse
        buf747 = reinterpret_tensor(buf546, (512, 512, 784), (401408, 784, 1), 0); del buf546  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_51.run(buf677, buf732, buf737, buf743, buf747, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf739, (512, 512, 784), (401408, 784, 1), 0), buf743, buf741, buf742, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf746 = buf743; del buf743  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf739, (512, 512, 784), (401408, 784, 1), 0), buf747, rsqrt_17, primals_106, buf741, buf742, buf746, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_106
        del rsqrt_17
        buf749 = reinterpret_tensor(buf725, (200704, 256), (256, 1), 0); del buf725  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_217, getitem_218, getitem_219, buf749, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_217
        del getitem_218
        del getitem_219
        buf751 = buf733; del buf733  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf752 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf746, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf751, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_18
        buf753 = buf752[0]
        assert_size_stride(buf753, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf753, 16, 'torch.ops.aten.convolution_backward.default')
        del buf752
        buf754 = buf751; del buf751  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf755 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf746, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf749, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf754, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf756 = buf755[1]
        assert_size_stride(buf756, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf756, 16, 'torch.ops.aten.convolution_backward.default')
        del buf755
        buf757 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(buf756, buf757, 65536, stream=stream0)
        del buf756
        buf758 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf779 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf5, buf758, buf779, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_216, reinterpret_tensor(buf758, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_216
        buf760 = buf749; del buf749  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_211, getitem_212, getitem_213, buf760, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_211
        del getitem_212
        del getitem_213
        buf762 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf763 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf783 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf784 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_46.run(buf636, buf762, buf763, buf783, buf784, 128, stream=stream0)
        buf764 = buf726; del buf726  # reuse
        buf768 = reinterpret_tensor(buf718, (512, 128, 784), (100352, 784, 1), 0); del buf718  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf753, buf758, buf764, buf768, 65536, 784, stream=stream0)
        del buf753
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf760, (512, 128, 784), (100352, 784, 1), 0), buf764, buf762, buf763, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf767 = buf764; del buf764  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf760, (512, 128, 784), (100352, 784, 1), 0), buf768, rsqrt_16, primals_100, buf762, buf763, buf767, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_100
        del rsqrt_16
        buf770 = reinterpret_tensor(buf768, (200704, 256), (256, 1), 0); del buf768  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_204, getitem_205, getitem_206, buf770, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_204
        del getitem_205
        del getitem_206
        buf772 = buf754; del buf754  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf773 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf767, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf772, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_17, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_17
        buf774 = buf773[0]
        assert_size_stride(buf774, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf774, 16, 'torch.ops.aten.convolution_backward.default')
        del buf773
        buf775 = buf772; del buf772  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf776 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf767, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf770, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf775, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf777 = buf776[1]
        assert_size_stride(buf777, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf777, 16, 'torch.ops.aten.convolution_backward.default')
        del buf776
        buf778 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_48.run(buf777, buf778, 147456, stream=stream0)
        del buf777
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_203, reinterpret_tensor(buf779, (200704, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del getitem_203
        buf781 = buf770; del buf770  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_198, getitem_199, getitem_200, buf781, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_198
        del getitem_199
        del getitem_200
        buf785 = buf767; del buf767  # reuse
        buf789 = reinterpret_tensor(buf760, (512, 128, 784), (100352, 784, 1), 0); del buf760  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf774, buf779, buf785, buf789, 65536, 784, stream=stream0)
        del buf774
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf781, (512, 128, 784), (100352, 784, 1), 0), buf785, buf783, buf784, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf788 = buf785; del buf785  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf781, (512, 128, 784), (100352, 784, 1), 0), buf789, rsqrt_15, primals_94, buf783, buf784, buf788, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_94
        del rsqrt_15
        buf791 = reinterpret_tensor(buf746, (802816, 256), (256, 1), 0); del buf746  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_191, getitem_192, getitem_193, buf791, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_191
        del getitem_192
        del getitem_193
        buf793 = buf775; del buf775  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf794 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf788, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf793, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_16
        buf795 = buf794[0]
        assert_size_stride(buf795, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf795, 16, 'torch.ops.aten.convolution_backward.default')
        del buf794
        buf796 = buf793; del buf793  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf797 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf788, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf791, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf796, (128, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf798 = buf797[1]
        assert_size_stride(buf798, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf798, 16, 'torch.ops.aten.convolution_backward.default')
        del buf797
        buf799 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(buf798, buf799, 65536, stream=stream0)
        del buf798
        buf800 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_41.run(buf610, buf800, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_190, reinterpret_tensor(buf800, (802816, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 802816, 1, 1, stream=stream0)
        del getitem_190
        buf802 = buf791; del buf791  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_185, getitem_186, getitem_187, buf802, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_185
        del getitem_186
        del getitem_187
        buf804 = buf732; del buf732  # reuse
        buf807 = buf747; del buf747  # reuse
        buf811 = reinterpret_tensor(buf739, (512, 512, 784), (401408, 784, 1), 0); del buf739  # reuse
        buf825 = buf621; del buf621  # reuse
        buf829 = reinterpret_tensor(buf613, (512, 512, 784), (401408, 784, 1), 0); del buf613  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_52.run(buf804, buf677, buf737, buf795, buf800, buf807, buf811, buf825, buf829, 262144, 784, stream=stream0)
        del buf677
        del buf795
        del buf804
        buf805 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf806 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf824 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_53.run(buf33, buf805, buf806, buf824, 512, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf802, (512, 512, 784), (401408, 784, 1), 0), buf807, buf805, buf806, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf810 = buf807; del buf807  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf802, (512, 512, 784), (401408, 784, 1), 0), buf811, rsqrt_14, primals_88, buf805, buf806, buf810, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del buf802
        del buf811
        del primals_88
        del rsqrt_14
        buf813 = empty_strided_cuda((1605632, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_178, getitem_179, getitem_180, buf813, 16, 1, 256, 1, 2, 16, 16, 1605632, 1, 1, stream=stream0)
        del getitem_178
        del getitem_179
        del getitem_180
        buf815 = buf796; del buf796  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf816 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf810, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf815, (512, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_15, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_15
        buf817 = buf816[0]
        assert_size_stride(buf817, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf817, 16, 'torch.ops.aten.convolution_backward.default')
        del buf816
        buf818 = buf815; del buf815  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf819 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf810, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf813, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf818, (512, 256, 1, 1), (0, 0, 0, 0), 0), None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf820 = buf819[1]
        assert_size_stride(buf820, (512, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf820, 16, 'torch.ops.aten.convolution_backward.default')
        del buf819
        buf821 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_39.run(buf820, buf821, 131072, stream=stream0)
        del buf820
        buf822 = reinterpret_tensor(buf810, (802816, 256), (256, 1), 0); del buf810  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_175, getitem_176, getitem_177, buf822, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_175
        del getitem_176
        del getitem_177
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf822, (512, 512, 784), (401408, 784, 1), 0), buf825, buf33, buf824, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf828 = buf825; del buf825  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf822, (512, 512, 784), (401408, 784, 1), 0), buf829, rsqrt_13, primals_82, buf33, buf824, buf828, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_82
        del rsqrt_13
        buf831 = reinterpret_tensor(buf788, (200704, 256), (256, 1), 0); del buf788  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_168, getitem_169, getitem_170, buf831, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_168
        del getitem_169
        del getitem_170
        buf833 = buf818; del buf818  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf834 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf828, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf833, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_14, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_14
        buf835 = buf834[0]
        assert_size_stride(buf835, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf835, 16, 'torch.ops.aten.convolution_backward.default')
        del buf834
        buf836 = buf833; del buf833  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf837 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf828, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf831, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf836, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf838 = buf837[1]
        assert_size_stride(buf838, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf838, 16, 'torch.ops.aten.convolution_backward.default')
        del buf837
        buf839 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(buf838, buf839, 65536, stream=stream0)
        del buf838
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_167, buf5, 8, 1, 256, 1, 1, 32, 8, 200704, 1, 1, stream=stream0)
        del buf135
        del buf191
        del buf6
        del buf632
        del buf654
        del buf695
        del buf716
        del buf72
        del buf758
        del buf779
        del getitem_167
        buf841 = buf831; del buf831  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_162, getitem_163, getitem_164, buf841, 16, 1, 256, 1, 2, 16, 16, 200704, 1, 1, stream=stream0)
        del getitem_162
        del getitem_163
        del getitem_164
        buf843 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf844 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf863 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_54.run(buf636, buf843, buf844, buf863, 128, stream=stream0)
        buf845 = buf789; del buf789  # reuse
        buf849 = reinterpret_tensor(buf781, (512, 128, 784), (100352, 784, 1), 0); del buf781  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf835, buf5, buf845, buf849, 65536, 784, stream=stream0)
        del buf5
        del buf835
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf841, (512, 128, 784), (100352, 784, 1), 0), buf845, buf843, buf844, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf848 = buf845; del buf845  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf841, (512, 128, 784), (100352, 784, 1), 0), buf849, rsqrt_12, primals_76, buf843, buf844, buf848, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del buf841
        del buf849
        del primals_76
        del rsqrt_12
        buf851 = reinterpret_tensor(buf828, (802816, 256), (256, 1), 0); del buf828  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_155, getitem_156, getitem_157, buf851, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_155
        del getitem_156
        del getitem_157
        buf853 = buf836; del buf836  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf854 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf848, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf853, (512, 128, 56, 56), (0, 0, 0, 0), 0), convert_element_type_13, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_13
        buf855 = buf854[0]
        assert_size_stride(buf855, (512, 128, 56, 56), (401408, 1, 7168, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf855, 16, 'torch.ops.aten.convolution_backward.default')
        del buf854
        buf856 = buf853; del buf853  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf857 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf848, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf851, (512, 128, 56, 56), (401408, 3136, 56, 1), 0), reinterpret_tensor(buf856, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        del buf848
        buf858 = buf857[1]
        assert_size_stride(buf858, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf858, 16, 'torch.ops.aten.convolution_backward.default')
        del buf857
        buf859 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_48.run(buf858, buf859, 147456, stream=stream0)
        del buf858
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_154, buf610, 8, 1, 256, 1, 1, 32, 8, 802816, 1, 1, stream=stream0)
        del buf611
        del buf675
        del buf737
        del buf800
        del getitem_154
        buf861 = buf851; del buf851  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_149, getitem_150, getitem_151, buf861, 16, 1, 256, 1, 2, 16, 16, 802816, 1, 1, stream=stream0)
        del getitem_149
        del getitem_150
        del getitem_151
        buf864 = reinterpret_tensor(buf829, (512, 128, 3136), (401408, 3136, 1), 0); del buf829  # reuse
        buf868 = reinterpret_tensor(buf822, (512, 128, 3136), (401408, 3136, 1), 0); del buf822  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_55.run(buf855, buf610, buf864, buf868, 65536, 3136, stream=stream0)
        del buf610
        del buf855
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_18.run(reinterpret_tensor(buf861, (512, 128, 3136), (401408, 3136, 1), 0), buf864, buf636, buf863, 1605632, 3136, 401408, 3136, 1024, 128, 1568, 1, stream=stream0)
        buf867 = buf864; del buf864  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_19.run(reinterpret_tensor(buf861, (512, 128, 3136), (401408, 3136, 1), 0), buf868, rsqrt_11, primals_70, buf636, buf863, buf867, 1605632, 3136, 401408, 3136, 1024, 128, 1568, 1, stream=stream0)
        del buf861
        del buf868
        del primals_70
        del rsqrt_11
        buf870 = buf813; del buf813  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_142, getitem_143, getitem_144, buf870, 16, 1, 256, 1, 2, 16, 16, 1605632, 1, 1, stream=stream0)
        del getitem_142
        del getitem_143
        del getitem_144
        buf872 = buf856; del buf856  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf873 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf867, (512, 128, 56, 56), (401408, 3136, 56, 1), 0), reinterpret_tensor(buf872, (512, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_12
        buf874 = buf873[0]
        assert_size_stride(buf874, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf874, 16, 'torch.ops.aten.convolution_backward.default')
        del buf873
        buf875 = buf872; del buf872  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf876 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf867, (512, 128, 56, 56), (401408, 3136, 56, 1), 0), reinterpret_tensor(buf870, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf875, (128, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        del buf867
        buf877 = buf876[1]
        assert_size_stride(buf877, (128, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf877, 16, 'torch.ops.aten.convolution_backward.default')
        del buf876
        buf878 = empty_strided_cuda((128, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_56.run(buf877, buf878, 32768, stream=stream0)
        del buf877
        buf879 = empty_strided_cuda((1605632, 256), (256, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_380], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_57.run(buf879, 411041792, stream=stream0)
        buf880 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_58.run(buf879, buf880, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_141, reinterpret_tensor(buf880, (1605632, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 1605632, 1, 1, stream=stream0)
        del getitem_141
        buf882 = buf870; del buf870  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_136, getitem_137, getitem_138, buf882, 16, 1, 256, 1, 2, 16, 16, 1605632, 1, 1, stream=stream0)
        del getitem_136
        del getitem_137
        del getitem_138
        buf884 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf885 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf949 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf950 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_28.run(buf240, buf884, buf885, buf949, buf950, 256, stream=stream0)
        buf886 = empty_strided_cuda((512, 256, 3136), (802816, 3136, 1), torch.bfloat16)
        buf890 = empty_strided_cuda((512, 256, 3136), (802816, 3136, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_59.run(buf817, buf874, buf880, buf886, buf890, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf882, (512, 256, 3136), (802816, 3136, 1), 0), buf886, buf884, buf885, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf889 = buf886; del buf886  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf882, (512, 256, 3136), (802816, 3136, 1), 0), buf890, rsqrt_10, primals_64, buf884, buf885, buf889, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_64
        del rsqrt_10
        buf892 = reinterpret_tensor(buf598, (401408, 256), (256, 1), 0); del buf598  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_129, getitem_130, getitem_131, buf892, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_129
        del getitem_130
        del getitem_131
        buf894 = buf875; del buf875  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf895 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf889, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf894, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_11, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_11
        buf896 = buf895[0]
        assert_size_stride(buf896, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf896, 16, 'torch.ops.aten.convolution_backward.default')
        del buf895
        buf897 = buf894; del buf894  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf898 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf889, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf892, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf897, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf899 = buf898[1]
        assert_size_stride(buf899, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf899, 16, 'torch.ops.aten.convolution_backward.default')
        del buf898
        buf900 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_60.run(buf899, buf900, 16384, stream=stream0)
        del buf899
        buf901 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf923 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_31.run(buf212, buf901, buf923, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_128, reinterpret_tensor(buf901, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_128
        buf903 = buf892; del buf892  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_123, getitem_124, getitem_125, buf903, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_123
        del getitem_124
        del getitem_125
        buf905 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_61.run(buf905, 64, stream=stream0)
        buf906 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf907 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf927 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf928 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_62.run(buf905, buf906, buf907, buf927, buf928, 64, stream=stream0)
        buf908 = reinterpret_tensor(buf599, (512, 64, 3136), (200704, 3136, 1), 0); del buf599  # reuse
        buf912 = reinterpret_tensor(buf591, (512, 64, 3136), (200704, 3136, 1), 0); del buf591  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_63.run(buf896, buf901, buf908, buf912, 32768, 3136, stream=stream0)
        del buf896
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf903, (512, 64, 3136), (200704, 3136, 1), 0), buf908, buf906, buf907, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf911 = buf908; del buf908  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf903, (512, 64, 3136), (200704, 3136, 1), 0), buf912, rsqrt_9, primals_58, buf906, buf907, buf911, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_58
        del rsqrt_9
        buf914 = reinterpret_tensor(buf912, (401408, 256), (256, 1), 0); del buf912  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_116, getitem_117, getitem_118, buf914, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_116
        del getitem_117
        del getitem_118
        buf916 = buf897; del buf897  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf917 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf911, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf916, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_10, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_10
        buf918 = buf917[0]
        assert_size_stride(buf918, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf918, 16, 'torch.ops.aten.convolution_backward.default')
        del buf917
        buf919 = buf916; del buf916  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf920 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf911, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf914, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf919, (64, 64, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf921 = buf920[1]
        assert_size_stride(buf921, (64, 64, 3, 3), (576, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf921, 16, 'torch.ops.aten.convolution_backward.default')
        del buf920
        buf922 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_64.run(buf921, buf922, 36864, stream=stream0)
        del buf921
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_115, reinterpret_tensor(buf923, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_115
        buf925 = buf914; del buf914  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_110, getitem_111, getitem_112, buf925, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_110
        del getitem_111
        del getitem_112
        buf929 = buf911; del buf911  # reuse
        buf933 = reinterpret_tensor(buf903, (512, 64, 3136), (200704, 3136, 1), 0); del buf903  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_63.run(buf918, buf923, buf929, buf933, 32768, 3136, stream=stream0)
        del buf918
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf925, (512, 64, 3136), (200704, 3136, 1), 0), buf929, buf927, buf928, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf932 = buf929; del buf929  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf925, (512, 64, 3136), (200704, 3136, 1), 0), buf933, rsqrt_8, primals_52, buf927, buf928, buf932, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_52
        del rsqrt_8
        buf935 = reinterpret_tensor(buf889, (1605632, 256), (256, 1), 0); del buf889  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_103, getitem_104, getitem_105, buf935, 16, 1, 256, 1, 2, 16, 16, 1605632, 1, 1, stream=stream0)
        del getitem_103
        del getitem_104
        del getitem_105
        buf937 = buf919; del buf919  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf938 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf932, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf937, (512, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_9
        buf939 = buf938[0]
        assert_size_stride(buf939, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf939, 16, 'torch.ops.aten.convolution_backward.default')
        del buf938
        buf940 = buf937; del buf937  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf941 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf932, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf935, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf940, (64, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf942 = buf941[1]
        assert_size_stride(buf942, (64, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf942, 16, 'torch.ops.aten.convolution_backward.default')
        del buf941
        buf943 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_60.run(buf942, buf943, 16384, stream=stream0)
        del buf942
        buf944 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        buf1006 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_65.run(buf879, buf944, buf1006, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_102, reinterpret_tensor(buf944, (1605632, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 1605632, 1, 1, stream=stream0)
        del getitem_102
        buf946 = reinterpret_tensor(buf935, (512, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf935  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_66.run(buf817, buf874, buf880, buf939, buf944, buf946, 131072, 3136, stream=stream0)
        buf947 = reinterpret_tensor(buf939, (1605632, 256), (256, 1), 0); del buf939  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_97, getitem_98, getitem_99, buf947, 16, 1, 256, 1, 2, 16, 16, 1605632, 1, 1, stream=stream0)
        del getitem_97
        del getitem_98
        del getitem_99
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf947, (512, 256, 3136), (802816, 3136, 1), 0), reinterpret_tensor(buf946, (512, 256, 3136), (802816, 3136, 1), 0), buf949, buf950, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf953 = reinterpret_tensor(buf874, (512, 256, 3136), (802816, 3136, 1), 0); del buf874  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf947, (512, 256, 3136), (802816, 3136, 1), 0), reinterpret_tensor(buf946, (512, 256, 3136), (802816, 3136, 1), 0), rsqrt_7, primals_46, buf949, buf950, buf953, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_46
        del rsqrt_7
        buf955 = reinterpret_tensor(buf932, (401408, 256), (256, 1), 0); del buf932  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_90, getitem_91, getitem_92, buf955, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_90
        del getitem_91
        del getitem_92
        buf957 = buf940; del buf940  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf958 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf953, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf957, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_8, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_8
        buf959 = buf958[0]
        assert_size_stride(buf959, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf959, 16, 'torch.ops.aten.convolution_backward.default')
        del buf958
        buf960 = buf957; del buf957  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf961 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf953, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf955, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf960, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf962 = buf961[1]
        assert_size_stride(buf962, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf962, 16, 'torch.ops.aten.convolution_backward.default')
        del buf961
        buf963 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_60.run(buf962, buf963, 16384, stream=stream0)
        del buf962
        buf964 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf985 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_31.run(buf212, buf964, buf985, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_89, reinterpret_tensor(buf964, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_89
        buf966 = buf955; del buf955  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_84, getitem_85, getitem_86, buf966, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_84
        del getitem_85
        del getitem_86
        buf968 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf969 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf989 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf990 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_62.run(buf905, buf968, buf969, buf989, buf990, 64, stream=stream0)
        buf970 = buf933; del buf933  # reuse
        buf974 = reinterpret_tensor(buf925, (512, 64, 3136), (200704, 3136, 1), 0); del buf925  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_63.run(buf959, buf964, buf970, buf974, 32768, 3136, stream=stream0)
        del buf959
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf966, (512, 64, 3136), (200704, 3136, 1), 0), buf970, buf968, buf969, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf973 = buf970; del buf970  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf966, (512, 64, 3136), (200704, 3136, 1), 0), buf974, rsqrt_6, primals_40, buf968, buf969, buf973, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_40
        del rsqrt_6
        buf976 = reinterpret_tensor(buf974, (401408, 256), (256, 1), 0); del buf974  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_77, getitem_78, getitem_79, buf976, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_77
        del getitem_78
        del getitem_79
        buf978 = buf960; del buf960  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf979 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf973, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf978, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_7
        buf980 = buf979[0]
        assert_size_stride(buf980, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf980, 16, 'torch.ops.aten.convolution_backward.default')
        del buf979
        buf981 = buf978; del buf978  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf982 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf973, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf976, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf981, (64, 64, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf983 = buf982[1]
        assert_size_stride(buf983, (64, 64, 3, 3), (576, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf983, 16, 'torch.ops.aten.convolution_backward.default')
        del buf982
        buf984 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_64.run(buf983, buf984, 36864, stream=stream0)
        del buf983
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_76, reinterpret_tensor(buf985, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_76
        buf987 = buf976; del buf976  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_71, getitem_72, getitem_73, buf987, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_71
        del getitem_72
        del getitem_73
        buf991 = buf973; del buf973  # reuse
        buf995 = reinterpret_tensor(buf966, (512, 64, 3136), (200704, 3136, 1), 0); del buf966  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_63.run(buf980, buf985, buf991, buf995, 32768, 3136, stream=stream0)
        del buf980
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf987, (512, 64, 3136), (200704, 3136, 1), 0), buf991, buf989, buf990, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf994 = buf991; del buf991  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf987, (512, 64, 3136), (200704, 3136, 1), 0), buf995, rsqrt_5, primals_34, buf989, buf990, buf994, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_34
        del rsqrt_5
        buf997 = reinterpret_tensor(buf953, (1605632, 256), (256, 1), 0); del buf953  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_64, getitem_65, getitem_66, buf997, 16, 1, 256, 1, 2, 16, 16, 1605632, 1, 1, stream=stream0)
        del getitem_64
        del getitem_65
        del getitem_66
        buf999 = buf981; del buf981  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1000 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf994, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf999, (512, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_6
        buf1001 = buf1000[0]
        assert_size_stride(buf1001, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1001, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1000
        buf1002 = buf999; del buf999  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1003 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf994, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf997, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1002, (64, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1004 = buf1003[1]
        assert_size_stride(buf1004, (64, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1004, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1003
        buf1005 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_60.run(buf1004, buf1005, 16384, stream=stream0)
        del buf1004
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_63, reinterpret_tensor(buf1006, (1605632, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 1605632, 1, 1, stream=stream0)
        del getitem_63
        buf1008 = buf997; del buf997  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_58, getitem_59, getitem_60, buf1008, 16, 1, 256, 1, 2, 16, 16, 1605632, 1, 1, stream=stream0)
        del getitem_58
        del getitem_59
        del getitem_60
        buf1010 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1011 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1029 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_67.run(buf240, buf1010, buf1011, buf1029, 256, stream=stream0)
        buf1012 = reinterpret_tensor(buf947, (512, 256, 3136), (802816, 3136, 1), 0); del buf947  # reuse
        buf1016 = reinterpret_tensor(buf817, (512, 256, 3136), (802816, 3136, 1), 0); del buf817  # reuse
        buf1030 = buf890; del buf890  # reuse
        buf1034 = reinterpret_tensor(buf882, (512, 256, 3136), (802816, 3136, 1), 0); del buf882  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf946, buf1001, buf1006, buf1012, buf1016, buf1030, buf1034, 131072, 3136, stream=stream0)
        del buf1001
        del buf946
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf1008, (512, 256, 3136), (802816, 3136, 1), 0), buf1012, buf1010, buf1011, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf1015 = buf1012; del buf1012  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf1008, (512, 256, 3136), (802816, 3136, 1), 0), buf1016, rsqrt_4, primals_28, buf1010, buf1011, buf1015, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del buf1008
        del primals_28
        del rsqrt_4
        buf1018 = reinterpret_tensor(buf994, (401408, 256), (256, 1), 0); del buf994  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_51, getitem_52, getitem_53, buf1018, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_51
        del getitem_52
        del getitem_53
        buf1020 = buf1002; del buf1002  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1021 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1015, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1020, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_5
        buf1022 = buf1021[0]
        assert_size_stride(buf1022, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1022, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1021
        buf1023 = buf1020; del buf1020  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1024 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1015, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1018, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1023, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1025 = buf1024[1]
        assert_size_stride(buf1025, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1025, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1024
        buf1026 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_60.run(buf1025, buf1026, 16384, stream=stream0)
        del buf1025
        buf1027 = reinterpret_tensor(buf1015, (1605632, 256), (256, 1), 0); del buf1015  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_48, getitem_49, getitem_50, buf1027, 16, 1, 256, 1, 2, 16, 16, 1605632, 1, 1, stream=stream0)
        del getitem_48
        del getitem_49
        del getitem_50
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf1027, (512, 256, 3136), (802816, 3136, 1), 0), buf1030, buf240, buf1029, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf1033 = buf1030; del buf1030  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf1027, (512, 256, 3136), (802816, 3136, 1), 0), buf1034, rsqrt_3, primals_22, buf240, buf1029, buf1033, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_22
        del rsqrt_3
        buf1036 = buf1018; del buf1018  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_41, getitem_42, getitem_43, buf1036, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_41
        del getitem_42
        del getitem_43
        buf1038 = buf1023; del buf1023  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1039 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1033, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1038, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_4, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_4
        buf1040 = buf1039[0]
        assert_size_stride(buf1040, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1040, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1039
        buf1041 = buf1038; del buf1038  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1042 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1033, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1036, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1041, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1043 = buf1042[1]
        assert_size_stride(buf1043, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1043, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1042
        buf1044 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_60.run(buf1043, buf1044, 16384, stream=stream0)
        del buf1043
        buf1045 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_20.run(buf212, buf1045, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_40, reinterpret_tensor(buf1045, (401408, 256), (256, 1), 0), 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del getitem_40
        buf1047 = buf1036; del buf1036  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_35, getitem_36, getitem_37, buf1047, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_35
        del getitem_36
        del getitem_37
        buf1049 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1050 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1069 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1070 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1091 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_69.run(buf905, buf1049, buf1050, buf1069, buf1070, buf1091, 64, stream=stream0)
        buf1051 = buf995; del buf995  # reuse
        buf1055 = reinterpret_tensor(buf987, (512, 64, 3136), (200704, 3136, 1), 0); del buf987  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_63.run(buf1040, buf1045, buf1051, buf1055, 32768, 3136, stream=stream0)
        del buf1040
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf1047, (512, 64, 3136), (200704, 3136, 1), 0), buf1051, buf1049, buf1050, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf1054 = buf1051; del buf1051  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf1047, (512, 64, 3136), (200704, 3136, 1), 0), buf1055, rsqrt_2, primals_16, buf1049, buf1050, buf1054, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_16
        del rsqrt_2
        buf1057 = reinterpret_tensor(buf1055, (401408, 256), (256, 1), 0); del buf1055  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_28, getitem_29, getitem_30, buf1057, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_28
        del getitem_29
        del getitem_30
        buf1059 = buf1041; del buf1041  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1060 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1054, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1059, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_3, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_3
        buf1061 = buf1060[0]
        assert_size_stride(buf1061, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1061, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1060
        buf1062 = buf1059; del buf1059  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1063 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1054, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1057, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1062, (64, 64, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf1064 = buf1063[1]
        assert_size_stride(buf1064, (64, 64, 3, 3), (576, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1064, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1063
        buf1065 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_64.run(buf1064, buf1065, 36864, stream=stream0)
        del buf1064
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_27, buf212, 8, 1, 256, 1, 1, 32, 8, 401408, 1, 1, stream=stream0)
        del buf1045
        del buf213
        del buf279
        del buf341
        del buf404
        del buf466
        del buf529
        del buf589
        del buf901
        del buf923
        del buf964
        del buf985
        del getitem_27
        buf1067 = buf1057; del buf1057  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_22, getitem_23, getitem_24, buf1067, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_22
        del getitem_23
        del getitem_24
        buf1071 = buf1054; del buf1054  # reuse
        buf1075 = reinterpret_tensor(buf1047, (512, 64, 3136), (200704, 3136, 1), 0); del buf1047  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_63.run(buf1061, buf212, buf1071, buf1075, 32768, 3136, stream=stream0)
        del buf1061
        del buf212
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf1067, (512, 64, 3136), (200704, 3136, 1), 0), buf1071, buf1069, buf1070, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf1074 = buf1071; del buf1071  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf1067, (512, 64, 3136), (200704, 3136, 1), 0), buf1075, rsqrt_1, primals_10, buf1069, buf1070, buf1074, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del buf1067
        del primals_10
        del rsqrt_1
        buf1077 = reinterpret_tensor(buf1075, (401408, 256), (256, 1), 0); del buf1075  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_15, getitem_16, getitem_17, buf1077, 16, 1, 256, 1, 2, 16, 16, 401408, 1, 1, stream=stream0)
        del getitem_15
        del getitem_16
        del getitem_17
        buf1079 = buf1062; del buf1062  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1080 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1074, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1079, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_2, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_2
        buf1081 = buf1080[0]
        assert_size_stride(buf1081, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1081, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1080
        buf1082 = buf1079; del buf1079  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1083 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1074, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1077, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1082, (64, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        del buf1074
        del buf1077
        buf1084 = buf1083[1]
        assert_size_stride(buf1084, (64, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1084, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1083
        buf1085 = empty_strided_cuda((64, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_70.run(buf1084, buf1085, 4096, stream=stream0)
        del buf1084
        buf1086 = buf1022; del buf1022  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_71.run(buf1086, buf1081, 102760448, stream=stream0)
        del buf1081
        buf1087 = reinterpret_tensor(buf1033, (512, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf1033  # reuse
        # Topologically Sorted Source Nodes: [maxpool], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_72.run(getitem_14, buf1086, buf1087, 6422528, 64, stream=stream0)
        del buf1086
        del getitem_14
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_12, buf879, 8, 1, 256, 1, 1, 32, 8, 1605632, 1, 1, stream=stream0)
        del buf1006
        del buf880
        del buf944
        del getitem_12
        buf1089 = reinterpret_tensor(buf1034, (1605632, 256), (256, 1), 0); del buf1034  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_7, getitem_8, getitem_9, buf1089, 16, 1, 256, 1, 2, 16, 16, 1605632, 1, 1, stream=stream0)
        del getitem_7
        del getitem_8
        del getitem_9
        buf1092 = reinterpret_tensor(buf1027, (512, 64, 12544), (802816, 12544, 1), 0); del buf1027  # reuse
        buf1096 = reinterpret_tensor(buf1016, (512, 64, 12544), (802816, 12544, 1), 0); del buf1016  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_73.run(buf1087, buf879, buf1092, buf1096, 411041792, stream=stream0)
        del buf1087
        del buf879
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_24.run(reinterpret_tensor(buf1089, (512, 64, 12544), (802816, 12544, 1), 0), buf1092, buf905, buf1091, 6422528, 12544, 802816, 12544, 1024, 64, 6272, 1, stream=stream0)
        buf1095 = buf1092; del buf1092  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_25.run(reinterpret_tensor(buf1089, (512, 64, 12544), (802816, 12544, 1), 0), buf1096, rsqrt, primals_4, buf905, buf1091, buf1095, 6422528, 12544, 802816, 12544, 1024, 64, 6272, 1, stream=stream0)
        del buf1089
        del buf1096
        del primals_4
        del rsqrt
        buf1098 = empty_strided_cuda((301056, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem, getitem_1, getitem_2, buf1098, 16, 1, 256, 1, 2, 16, 16, 301056, 1, 1, stream=stream0)
        del getitem
        del getitem_1
        del getitem_2
        buf1100 = buf1082; del buf1082  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1101 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1095, (512, 64, 112, 112), (802816, 12544, 112, 1), 0), reinterpret_tensor(buf1098, (512, 3, 224, 224), (150528, 50176, 224, 1), 0), reinterpret_tensor(buf1100, (64, 3, 7, 7), (0, 0, 0, 0), 0), None, [2, 2], [3, 3], [1, 1], False, [0], 1, [False, True, False])
        del buf1095
        del buf1098
        del buf1100
        buf1102 = buf1101[1]
        assert_size_stride(buf1102, (64, 3, 7, 7), (147, 49, 7, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1102, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1101
        buf1103 = empty_strided_cuda((64, 3, 7, 7), (147, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_74.run(buf1102, buf1103, 9408, stream=stream0)
        del buf1102
    return (buf1103, None, None, buf1091, buf905, None, None, buf1085, None, buf1070, buf1069, None, None, buf1065, None, buf1050, buf1049, None, None, buf1044, None, buf1029, buf240, None, None, buf1026, None, buf1011, buf1010, None, None, buf1005, None, buf990, buf989, None, None, buf984, None, buf969, buf968, None, None, buf963, None, buf950, buf949, None, None, buf943, None, buf928, buf927, None, None, buf922, None, buf907, buf906, None, None, buf900, None, buf885, buf884, None, None, buf878, None, buf863, buf636, None, None, buf859, None, buf844, buf843, None, None, buf839, None, buf824, buf33, None, None, buf821, None, buf806, buf805, None, None, buf799, None, buf784, buf783, None, None, buf778, None, buf763, buf762, None, None, buf757, None, buf742, buf741, None, None, buf736, None, buf721, buf720, None, None, buf715, None, buf700, buf699, None, None, buf694, None, buf681, buf680, None, None, buf674, None, buf659, buf658, None, None, buf653, None, buf638, buf637, None, None, buf631, None, buf616, buf615, None, None, buf609, None, buf594, buf593, None, None, buf588, None, buf573, buf572, None, None, buf568, None, buf553, buf217, None, None, buf550, None, buf535, buf534, None, None, buf528, None, buf513, buf512, None, None, buf507, None, buf492, buf491, None, None, buf486, None, buf471, buf470, None, None, buf465, None, buf450, buf449, None, None, buf444, None, buf429, buf428, None, None, buf423, None, buf410, buf409, None, None, buf403, None, buf388, buf387, None, None, buf382, None, buf367, buf366, None, None, buf361, None, buf346, buf345, None, None, buf340, None, buf325, buf324, None, None, buf319, None, buf304, buf303, None, None, buf298, None, buf285, buf284, None, None, buf278, None, buf263, buf262, None, None, buf257, None, buf242, buf241, None, None, buf234, None, buf219, buf218, None, None, buf211, None, buf196, buf195, None, None, buf190, None, buf175, buf174, None, None, buf170, None, buf157, buf10, None, None, buf154, None, buf141, buf140, None, None, buf134, None, buf119, buf118, None, None, buf113, None, buf98, buf97, None, None, buf92, None, buf77, buf76, None, None, buf71, None, buf56, buf55, None, None, buf50, None, buf35, buf34, None, None, buf27, None, buf12, buf11, None, None, buf4, )


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
    getitem = rand_strided((301056, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_1 = rand_strided((301056, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_2 = rand_strided((301056, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_8 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_10 = rand_strided((512, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_12 = rand_strided((1605632, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    getitem_14 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.int8)
    convert_element_type_2 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_15 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_16 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_17 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_22 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_23 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_24 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_27 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_3 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_28 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_29 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_30 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_36 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_37 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_40 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_4 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_41 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_42 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_43 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_48 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_49 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_50 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_5 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_51 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_52 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_53 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_59 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_60 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_63 = rand_strided((1605632, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_6 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_64 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_65 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_66 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_72 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_73 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_76 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_7 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_77 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_78 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_79 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_85 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_86 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_89 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_8 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_90 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_91 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_92 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_7 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_98 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_99 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_102 = rand_strided((1605632, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_9 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_103 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_104 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_105 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_110 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_111 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_112 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_115 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_10 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_116 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_117 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_118 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_124 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_125 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_128 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_11 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_129 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_130 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_131 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_136 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_137 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_138 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_141 = rand_strided((1605632, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_12 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_142 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_143 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_144 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_150 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_151 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_154 = rand_strided((802816, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_13 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_155 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_156 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_157 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_162 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_163 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_164 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_167 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_14 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_168 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_169 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_170 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_13 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_175 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_176 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_177 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_15 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_178 = rand_strided((1605632, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_179 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_180 = rand_strided((1605632, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_14 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_185 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_186 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_187 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_190 = rand_strided((802816, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_16 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_191 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_192 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_193 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_198 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_199 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_200 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_203 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_17 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_204 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_205 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_206 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_211 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_212 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_213 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_216 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_18 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_217 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_218 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_219 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_224 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_225 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_226 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_229 = rand_strided((802816, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_19 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_230 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_231 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_232 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_237 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_238 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_239 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_242 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_20 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_243 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_244 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_245 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_250 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_251 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_252 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_255 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_21 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_256 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_257 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_258 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_263 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_264 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_265 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_268 = rand_strided((802816, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_22 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_269 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_270 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_271 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_276 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_277 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_278 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_281 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_23 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_282 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_283 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_284 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_289 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_290 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_291 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_294 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_24 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_295 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_296 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_297 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_302 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_303 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_304 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_307 = rand_strided((802816, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_25 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_308 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_309 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_310 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_315 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_316 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_317 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_320 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_26 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_321 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_322 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_323 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_328 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_329 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_330 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_333 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_27 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_334 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_335 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_336 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_26 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_341 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_342 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_343 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_28 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_344 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_345 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_346 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_27 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_351 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_352 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_353 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_356 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_29 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_357 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_358 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_359 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_364 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_365 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_366 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_369 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_30 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_370 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_371 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_372 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_377 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_378 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_379 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_382 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_31 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_383 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_384 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_385 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_30 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_390 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_391 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_392 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_395 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_32 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_396 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_397 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_398 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_403 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_404 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_405 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_408 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_33 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_409 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_410 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_411 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_416 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_417 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_418 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_421 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_34 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_422 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_423 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_424 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_33 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_429 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_430 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_431 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_434 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_35 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_435 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_436 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_437 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_442 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_443 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_444 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_447 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_36 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_448 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_449 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_450 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_455 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_456 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_457 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_460 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_37 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_461 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_462 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_463 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_36 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_468 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_469 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_470 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_473 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_38 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_474 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_475 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_476 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_481 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_482 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_483 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_486 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_39 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_487 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_488 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_489 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_494 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_495 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_496 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_499 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_40 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_500 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_501 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_502 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_39 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_507 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_508 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_509 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_512 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_41 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_513 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_514 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_515 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_520 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_521 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_522 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_525 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_42 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_526 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_527 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_528 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_533 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_534 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_535 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_538 = rand_strided((100352, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_43 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_539 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_540 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_541 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_42 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_546 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_547 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_548 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_551 = rand_strided((401408, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_44 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_552 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_553 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_554 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_559 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_560 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_561 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_564 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_45 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_565 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_566 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_567 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_572 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_573 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_574 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_577 = rand_strided((50176, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_46 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_578 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_579 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_580 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_45 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_585 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_586 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_587 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_47 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_588 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_589 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_590 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_46 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_595 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_596 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_597 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_600 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_48 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.bfloat16)
    getitem_601 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_602 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_603 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_47 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_608 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_609 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_610 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_613 = rand_strided((50176, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_49 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_614 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_615 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_616 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_621 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_622 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_623 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_626 = rand_strided((50176, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_50 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_627 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_628 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_629 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_49 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_634 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_635 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_636 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_639 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_51 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.bfloat16)
    getitem_640 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_641 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_642 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_647 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_648 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_649 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_652 = rand_strided((50176, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_52 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_653 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_654 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_655 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_660 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_661 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_662 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_665 = rand_strided((50176, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_53 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_666 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_667 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_668 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_52 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_673 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_674 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_675 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_678 = rand_strided((200704, 8), (8, 1), device='cuda:0', dtype=torch.int32)
    getitem_679 = rand_strided((4096, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_680 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_681 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_54 = rand_strided((100, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((512, 100), (100, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, getitem, getitem_1, getitem_2, rsqrt, getitem_7, getitem_8, getitem_9, getitem_10, getitem_12, getitem_14, convert_element_type_2, getitem_15, getitem_16, getitem_17, rsqrt_1, getitem_22, getitem_23, getitem_24, getitem_27, convert_element_type_3, getitem_28, getitem_29, getitem_30, rsqrt_2, getitem_35, getitem_36, getitem_37, getitem_40, convert_element_type_4, getitem_41, getitem_42, getitem_43, rsqrt_3, getitem_48, getitem_49, getitem_50, convert_element_type_5, getitem_51, getitem_52, getitem_53, rsqrt_4, getitem_58, getitem_59, getitem_60, getitem_63, convert_element_type_6, getitem_64, getitem_65, getitem_66, rsqrt_5, getitem_71, getitem_72, getitem_73, getitem_76, convert_element_type_7, getitem_77, getitem_78, getitem_79, rsqrt_6, getitem_84, getitem_85, getitem_86, getitem_89, convert_element_type_8, getitem_90, getitem_91, getitem_92, rsqrt_7, getitem_97, getitem_98, getitem_99, getitem_102, convert_element_type_9, getitem_103, getitem_104, getitem_105, rsqrt_8, getitem_110, getitem_111, getitem_112, getitem_115, convert_element_type_10, getitem_116, getitem_117, getitem_118, rsqrt_9, getitem_123, getitem_124, getitem_125, getitem_128, convert_element_type_11, getitem_129, getitem_130, getitem_131, rsqrt_10, getitem_136, getitem_137, getitem_138, getitem_141, convert_element_type_12, getitem_142, getitem_143, getitem_144, rsqrt_11, getitem_149, getitem_150, getitem_151, getitem_154, convert_element_type_13, getitem_155, getitem_156, getitem_157, rsqrt_12, getitem_162, getitem_163, getitem_164, getitem_167, convert_element_type_14, getitem_168, getitem_169, getitem_170, rsqrt_13, getitem_175, getitem_176, getitem_177, convert_element_type_15, getitem_178, getitem_179, getitem_180, rsqrt_14, getitem_185, getitem_186, getitem_187, getitem_190, convert_element_type_16, getitem_191, getitem_192, getitem_193, rsqrt_15, getitem_198, getitem_199, getitem_200, getitem_203, convert_element_type_17, getitem_204, getitem_205, getitem_206, rsqrt_16, getitem_211, getitem_212, getitem_213, getitem_216, convert_element_type_18, getitem_217, getitem_218, getitem_219, rsqrt_17, getitem_224, getitem_225, getitem_226, getitem_229, convert_element_type_19, getitem_230, getitem_231, getitem_232, rsqrt_18, getitem_237, getitem_238, getitem_239, getitem_242, convert_element_type_20, getitem_243, getitem_244, getitem_245, rsqrt_19, getitem_250, getitem_251, getitem_252, getitem_255, convert_element_type_21, getitem_256, getitem_257, getitem_258, rsqrt_20, getitem_263, getitem_264, getitem_265, getitem_268, convert_element_type_22, getitem_269, getitem_270, getitem_271, rsqrt_21, getitem_276, getitem_277, getitem_278, getitem_281, convert_element_type_23, getitem_282, getitem_283, getitem_284, rsqrt_22, getitem_289, getitem_290, getitem_291, getitem_294, convert_element_type_24, getitem_295, getitem_296, getitem_297, rsqrt_23, getitem_302, getitem_303, getitem_304, getitem_307, convert_element_type_25, getitem_308, getitem_309, getitem_310, rsqrt_24, getitem_315, getitem_316, getitem_317, getitem_320, convert_element_type_26, getitem_321, getitem_322, getitem_323, rsqrt_25, getitem_328, getitem_329, getitem_330, getitem_333, convert_element_type_27, getitem_334, getitem_335, getitem_336, rsqrt_26, getitem_341, getitem_342, getitem_343, convert_element_type_28, getitem_344, getitem_345, getitem_346, rsqrt_27, getitem_351, getitem_352, getitem_353, getitem_356, convert_element_type_29, getitem_357, getitem_358, getitem_359, rsqrt_28, getitem_364, getitem_365, getitem_366, getitem_369, convert_element_type_30, getitem_370, getitem_371, getitem_372, rsqrt_29, getitem_377, getitem_378, getitem_379, getitem_382, convert_element_type_31, getitem_383, getitem_384, getitem_385, rsqrt_30, getitem_390, getitem_391, getitem_392, getitem_395, convert_element_type_32, getitem_396, getitem_397, getitem_398, rsqrt_31, getitem_403, getitem_404, getitem_405, getitem_408, convert_element_type_33, getitem_409, getitem_410, getitem_411, rsqrt_32, getitem_416, getitem_417, getitem_418, getitem_421, convert_element_type_34, getitem_422, getitem_423, getitem_424, rsqrt_33, getitem_429, getitem_430, getitem_431, getitem_434, convert_element_type_35, getitem_435, getitem_436, getitem_437, rsqrt_34, getitem_442, getitem_443, getitem_444, getitem_447, convert_element_type_36, getitem_448, getitem_449, getitem_450, rsqrt_35, getitem_455, getitem_456, getitem_457, getitem_460, convert_element_type_37, getitem_461, getitem_462, getitem_463, rsqrt_36, getitem_468, getitem_469, getitem_470, getitem_473, convert_element_type_38, getitem_474, getitem_475, getitem_476, rsqrt_37, getitem_481, getitem_482, getitem_483, getitem_486, convert_element_type_39, getitem_487, getitem_488, getitem_489, rsqrt_38, getitem_494, getitem_495, getitem_496, getitem_499, convert_element_type_40, getitem_500, getitem_501, getitem_502, rsqrt_39, getitem_507, getitem_508, getitem_509, getitem_512, convert_element_type_41, getitem_513, getitem_514, getitem_515, rsqrt_40, getitem_520, getitem_521, getitem_522, getitem_525, convert_element_type_42, getitem_526, getitem_527, getitem_528, rsqrt_41, getitem_533, getitem_534, getitem_535, getitem_538, convert_element_type_43, getitem_539, getitem_540, getitem_541, rsqrt_42, getitem_546, getitem_547, getitem_548, getitem_551, convert_element_type_44, getitem_552, getitem_553, getitem_554, rsqrt_43, getitem_559, getitem_560, getitem_561, getitem_564, convert_element_type_45, getitem_565, getitem_566, getitem_567, rsqrt_44, getitem_572, getitem_573, getitem_574, getitem_577, convert_element_type_46, getitem_578, getitem_579, getitem_580, rsqrt_45, getitem_585, getitem_586, getitem_587, convert_element_type_47, getitem_588, getitem_589, getitem_590, rsqrt_46, getitem_595, getitem_596, getitem_597, getitem_600, convert_element_type_48, getitem_601, getitem_602, getitem_603, rsqrt_47, getitem_608, getitem_609, getitem_610, getitem_613, convert_element_type_49, getitem_614, getitem_615, getitem_616, rsqrt_48, getitem_621, getitem_622, getitem_623, getitem_626, convert_element_type_50, getitem_627, getitem_628, getitem_629, rsqrt_49, getitem_634, getitem_635, getitem_636, getitem_639, convert_element_type_51, getitem_640, getitem_641, getitem_642, rsqrt_50, getitem_647, getitem_648, getitem_649, getitem_652, convert_element_type_52, getitem_653, getitem_654, getitem_655, rsqrt_51, getitem_660, getitem_661, getitem_662, getitem_665, convert_element_type_53, getitem_666, getitem_667, getitem_668, rsqrt_52, getitem_673, getitem_674, getitem_675, getitem_678, getitem_679, getitem_680, getitem_681, convert_element_type_54, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
