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


# kernel path: /tmp/torchinductor_yyu496/72/c726enjgfad5kdaz2spavtln2eeazaohbnnizdultkk3o55i6lk3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#    => constant_pad_nd_default
# Graph fragment:
#   %constant_pad_nd_default : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_156, [0, 0, 0, 4]), kwargs = {})
triton_poi_fused_mm_0 = async_compile.triton('triton_poi_fused_mm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 319488}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 104)
    x1 = xindex // 104
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 100, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 100*x1), tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gq/cgqkyytaim3ggm6djx7uqsszda5mt45vshjh3tync23mipbvtckw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_tensor, torch.float32), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vd/cvdqegtidjk6mvymbcn5lkrjihdgdtgh4mq6vyiou2emeezwwwjy.py
# Topologically Sorted Source Nodes: [full_257], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_257 => full_default_310
# Graph fragment:
#   %full_default_310 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([100352, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_2 = async_compile.triton('triton_poi_fused_zeros_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vx/cvxayt6kuzrkfr2hiooxyim2zjh67xwyqtaeskdrzz5r7drbykst.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_133, clone_default_142
# Graph fragment:
#   %clone_default_142 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_284,), kwargs = {})
#   %clone_default_133 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_266,), kwargs = {})
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256901120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
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


# kernel path: /tmp/torchinductor_yyu496/b3/cb3snufb3gt2neaopdf23lbjbsozri3k4vehlcqa5c3wzjdvsq3e.py
# Topologically Sorted Source Nodes: [layer4_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer4_0_bn3 => full_default_264
# Graph fragment:
#   %full_default_264 : [num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([2048], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_4 = async_compile.triton('triton_poi_fused_zeros_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16384}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/nd/cnd6shks5ykybmcoepeeeuh7uhy56gw5ofqmzvpmq7wafzxuoa7n.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_131, clone_default_132, clone_default_140, clone_default_141
# Graph fragment:
#   %clone_default_140 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_280,), kwargs = {})
#   %clone_default_141 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_282,), kwargs = {})
#   %clone_default_131 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_262,), kwargs = {})
#   %clone_default_132 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_264,), kwargs = {})
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 73728}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/y7/cy76axnx4nfllmbfpqhyuba6eyz3hjpayvgzj2d52r44eymf6rzh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_257, triton_kernel_wrapper_mutation_258
# Graph fragment:
#   %triton_kernel_wrapper_mutation_258 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 436, constant_args_idx: 625, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1351, DY: %view_1349, DBETA: %as_strided_default_281, DGAMMA: %as_strided_default_283, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_257 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 438, constant_args_idx: 626, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1351, DY: %view_1349, INVSTD: %rsqrt_52, GAMMA: %primals_316, DBETA: %as_strided_default_281, DGAMMA: %as_strided_default_283, DX: %permute_157, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/yq/cyqi7ypck54yyup5wyr7iu7xdzu3lr5j5uqjbgz4bvho36ukreuv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_67 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_693, torch.float32), kwargs = {})
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


# kernel path: /tmp/torchinductor_yyu496/er/cerbv2b7zjydhvfk7l43f5pfb6jznxiqdtcugjbuvpw3mc52elu2.py
# Topologically Sorted Source Nodes: [full_260], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_260 => full_default_313
# Graph fragment:
#   %full_default_313 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([25088, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_8 = async_compile.triton('triton_poi_fused_zeros_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25690112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4f/c4flau3zz2gfhbpbjg4cvoegjyuigjivtx2bxkq3bx27okl4u2lq.py
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
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 64225280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
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


# kernel path: /tmp/torchinductor_yyu496/ua/cuaee3khlrtviaemprir3jezlgco5bgenyuww5umneqxtjs2vffy.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_252, triton_kernel_wrapper_mutation_253
# Graph fragment:
#   %triton_kernel_wrapper_mutation_253 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 441, constant_args_idx: 630, grid: [(512, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1390, DY: %view_1388, DBETA: %as_strided_default_275, DGAMMA: %as_strided_default_277, M: 25088, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_252 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 442, constant_args_idx: 631, grid: [(512, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1390, DY: %view_1388, INVSTD: %rsqrt_51, GAMMA: %primals_310, DBETA: %as_strided_default_275, DGAMMA: %as_strided_default_277, DX: %permute_158, M: 25088, HW: 49, stride_n: 25088, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_12 = async_compile.triton('triton_poi_fused_12', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 25690112, 'x': 102760448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_12(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
#   %convert_element_type_72 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_705, torch.float32), kwargs = {})
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


# kernel path: /tmp/torchinductor_yyu496/t6/ct6qmzawdji5auqaq4uze2aeezefj6thcqacjbq7lpr5qavk3b7z.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_242, triton_kernel_wrapper_mutation_243
# Graph fragment:
#   %triton_kernel_wrapper_mutation_243 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 449, constant_args_idx: 640, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1468, DY: %view_1466, DBETA: %as_strided_default_263, DGAMMA: %as_strided_default_265, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_242 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 450, constant_args_idx: 641, grid: [(2048, 25, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1468, DY: %view_1466, INVSTD: %rsqrt_49, GAMMA: %primals_298, DBETA: %as_strided_default_263, DGAMMA: %as_strided_default_265, DX: %permute_160, M: 25088, HW: 49, stride_n: 100352, stride_c: 49, BLOCK_M: 1024}})
triton_poi_fused_14 = async_compile.triton('triton_poi_fused_14', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 104857600, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/ar/carxi4hp4dofhfrkbkn2e4ee4zxt2h46kbrgqeduwpfvz2j6bwec.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_159 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %mul_371 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_159, %view_1334), kwargs = {})
#   %add_228 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %getitem_713), kwargs = {})
#   %mul_374 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_228, %view_1451), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_374, %getitem_749), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_229, %view_1568), kwargs = {})
triton_poi_fused_add_div_mul_15 = async_compile.triton('triton_poi_fused_add_div_mul_15', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 207618048, 'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/fg/cfgieyjq2aafgww6haldblo7vmlzlgifcjxbheepxbp5jnbrq3mz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_121, clone_default_122, clone_default_123
# Graph fragment:
#   %clone_default_122 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_244,), kwargs = {})
#   %clone_default_123 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_246,), kwargs = {})
#   %clone_default_121 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_242,), kwargs = {})
triton_poi_fused_16 = async_compile.triton('triton_poi_fused_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 57344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/kd/ckd74x73hlebfoc5kkoyfywpio2jgqxnjiwlhqcrivlfvktxd7p4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_97 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_765, torch.float32), kwargs = {})
triton_poi_fused__to_copy_17 = async_compile.triton('triton_poi_fused__to_copy_17', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20971520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3h/c3hpiqjrucpi6j6n5c7xpjxs3gebtuvft3hx3mdcqr4glkaiiaf7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_213, triton_kernel_wrapper_mutation_214
# Graph fragment:
#   %triton_kernel_wrapper_mutation_214 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 473, constant_args_idx: 669, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1698, DY: %view_1696, DBETA: %as_strided_default_233, DGAMMA: %as_strided_default_235, M: 100352, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_213 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 474, constant_args_idx: 670, grid: [(512, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1698, DY: %view_1696, INVSTD: %rsqrt_43, GAMMA: %primals_262, DBETA: %as_strided_default_233, DGAMMA: %as_strided_default_235, DX: %permute_166, M: 100352, HW: 196, stride_n: 100352, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_18 = async_compile.triton('triton_poi_fused_18', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 102760448, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_18(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/ht/chthgpgdd2yh7yiteomd3w3epir3bxye7hobrcz4ptftpy6mocud.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_112 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_800, torch.float32), kwargs = {})
triton_poi_fused__to_copy_19 = async_compile.triton('triton_poi_fused__to_copy_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5242880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jt/cjthtnm7mnsgfriy6or54pfnxopbfbjr664vdgzk4rqgzmpin3pp.py
# Topologically Sorted Source Nodes: [full_286], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_286 => full_default_339
# Graph fragment:
#   %full_default_339 : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([200704, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_20 = async_compile.triton('triton_poi_fused_zeros_20', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_20(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/kb/ckbwmpyv2cz5ll2mygbnuso3wi34dpxxhmyshryzujscy7afq7iz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_115
# Graph fragment:
#   %clone_default_115 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_230,), kwargs = {})
triton_poi_fused_21 = async_compile.triton('triton_poi_fused_21', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 308281344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/d3/cd37qq3hejns7pcwds4ugmrwjwce66f2r6x7o2klbfjgxncv2suu.py
# Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer3_0_bn3 => full_default_152
# Graph fragment:
#   %full_default_152 : [num_users=16] = call_function[target=torch.ops.aten.full.default](args = ([1024], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_22 = async_compile.triton('triton_poi_fused_zeros_22', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_22(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sp/cspufs2nh226sqsbbbjif4uceuxdt7qrkoz6t45t6ehgpgmrg44u.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_113, clone_default_114
# Graph fragment:
#   %clone_default_113 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_226,), kwargs = {})
#   %clone_default_114 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_228,), kwargs = {})
triton_poi_fused_23 = async_compile.triton('triton_poi_fused_23', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_23(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/x5/cx57mb4palh2dznoct65gkwhl7uo6r2g3wdtek4vhc2z6ratumak.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_208, triton_kernel_wrapper_mutation_209
# Graph fragment:
#   %triton_kernel_wrapper_mutation_209 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 477, constant_args_idx: 674, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1737, DY: %view_1735, DBETA: %as_strided_default_227, DGAMMA: %as_strided_default_229, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_208 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 478, constant_args_idx: 675, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1737, DY: %view_1735, INVSTD: %rsqrt_42, GAMMA: %primals_256, DBETA: %as_strided_default_227, DGAMMA: %as_strided_default_229, DX: %permute_167, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_24 = async_compile.triton('triton_poi_fused_24', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/q7/cq7mrnxdhsqvrdh5iuuyi3hy72atk573z7vkneusrqt52tqq2sdg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_117 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_812, torch.float32), kwargs = {})
triton_poi_fused__to_copy_25 = async_compile.triton('triton_poi_fused__to_copy_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2621440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/su/csudzy35jcmy5kffdc63eqdwcogpptonbg3etosnc4ldbgqwow6r.py
# Topologically Sorted Source Nodes: [full_289], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_289 => full_default_342
# Graph fragment:
#   %full_default_342 : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([50176, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_26 = async_compile.triton('triton_poi_fused_zeros_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51380224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_26(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/kt/ckttcwsidtcsze7hrrogpepfcmht6ti5mqxdrqxmizwzjxh256wn.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_109, clone_default_112
# Graph fragment:
#   %clone_default_112 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_224,), kwargs = {})
#   %clone_default_109 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_218,), kwargs = {})
triton_poi_fused_27 = async_compile.triton('triton_poi_fused_27', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 128450560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_27(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/m6/cm6rtdj3pskys46uigbyo7z7ofmszzuiho7hbnynrqsn7hlomm5p.py
# Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer1_0_bn3 => full_default_18
# Graph fragment:
#   %full_default_18 : [num_users=34] = call_function[target=torch.ops.aten.full.default](args = ([256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_28 = async_compile.triton('triton_poi_fused_zeros_28', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_28(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ie/ciehuxxgggycgslwacghny66w52u6knsiff4wwa5qmffsmqppnkr.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_107, clone_default_108, clone_default_110, clone_default_111
# Graph fragment:
#   %clone_default_110 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_220,), kwargs = {})
#   %clone_default_111 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_222,), kwargs = {})
#   %clone_default_107 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_214,), kwargs = {})
#   %clone_default_108 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_216,), kwargs = {})
triton_poi_fused_29 = async_compile.triton('triton_poi_fused_29', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_29(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/rb/crbab3gtlcwsy54m6xqbhl3ph3jeqbigrgsjjjsk2raihzocxb44.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_203, triton_kernel_wrapper_mutation_204
# Graph fragment:
#   %triton_kernel_wrapper_mutation_204 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 481, constant_args_idx: 679, grid: [(256, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1776, DY: %view_1774, DBETA: %as_strided_default_221, DGAMMA: %as_strided_default_223, M: 100352, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_203 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 482, constant_args_idx: 680, grid: [(256, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1776, DY: %view_1774, INVSTD: %rsqrt_41, GAMMA: %primals_250, DBETA: %as_strided_default_221, DGAMMA: %as_strided_default_223, DX: %permute_168, M: 100352, HW: 196, stride_n: 50176, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_30 = async_compile.triton('triton_poi_fused_30', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 51380224, 'x': 205520896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_30(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/7i/c7isxg5ylfubjzsdupqziyrvnyhzjjfjwnms2ifbk2i2ca3nvia6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_122 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_824, torch.float32), kwargs = {})
triton_poi_fused__to_copy_31 = async_compile.triton('triton_poi_fused__to_copy_31', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5898240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/j7/cj7epf6bynt3h77mg22tyw2awsryghoapb5ty4lbcqnb3gvrcxyz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_106, clone_default_97
# Graph fragment:
#   %clone_default_106 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_212,), kwargs = {})
#   %clone_default_97 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_194,), kwargs = {})
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


# kernel path: /tmp/torchinductor_yyu496/zf/czfti4o5pcagwqqafmv75hosxzduyxvccvh3732xb2rpnxp74ced.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_230 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_761, %getitem_796), kwargs = {})
#   %mul_380 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, %view_1720), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %getitem_832), kwargs = {})
#   %mul_383 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_231, %view_1837), kwargs = {})
triton_poi_fused_add_mul_33 = async_compile.triton('triton_poi_fused_add_mul_33', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 616562688, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/z4/cz43kxr54rxrulm5ohcg5du4lybp5hkfxh57p3ks7jawo5bkqfkx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_104, clone_default_105, clone_default_95, clone_default_96
# Graph fragment:
#   %clone_default_104 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_208,), kwargs = {})
#   %clone_default_105 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_210,), kwargs = {})
#   %clone_default_95 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_190,), kwargs = {})
#   %clone_default_96 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_192,), kwargs = {})
triton_poi_fused_34 = async_compile.triton('triton_poi_fused_34', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_34(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/bu/cbuzareggsvctdwlwlhlbxzjjlqcjzyiqs7syzxbzttarf2k5hdj.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_178, triton_kernel_wrapper_mutation_179
# Graph fragment:
#   %triton_kernel_wrapper_mutation_179 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 501, constant_args_idx: 704, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1971, DY: %view_1969, DBETA: %as_strided_default_191, DGAMMA: %as_strided_default_193, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_178 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 502, constant_args_idx: 705, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_1971, DY: %view_1969, INVSTD: %rsqrt_36, GAMMA: %primals_220, DBETA: %as_strided_default_191, DGAMMA: %as_strided_default_193, DX: %permute_173, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i8', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 1027604480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_35(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/b3/cb3eaxc2s2jezingyn5zxoupicqt4exvbdqwllrwgxlcch7ipsnu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %getitem_868), kwargs = {})
#   %mul_386 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_232, %view_1954), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %getitem_904), kwargs = {})
#   %mul_389 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_233, %view_2071), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*i8', 'in_ptr2': '*bf16', 'in_ptr3': '*i8', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 616562688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/kr/ckrryqdt54nfyjnmhpf5p3jcv7wmuwptq6gaud57drmrpfz7u2if.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_129, triton_kernel_wrapper_mutation_130, triton_kernel_wrapper_mutation_133, triton_kernel_wrapper_mutation_134
# Graph fragment:
#   %add_234 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %getitem_940), kwargs = {})
#   %mul_392 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_234, %view_2188), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_392, %getitem_976), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_235, %view_2305), kwargs = {})
#   %triton_kernel_wrapper_mutation_134 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 537, constant_args_idx: 749, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2322, DY: %view_2320, DBETA: %as_strided_default_137, DGAMMA: %as_strided_default_139, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_133 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 538, constant_args_idx: 750, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2322, DY: %view_2320, INVSTD: %rsqrt_27, GAMMA: %primals_166, DBETA: %as_strided_default_137, DGAMMA: %as_strided_default_139, DX: %permute_182, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_130 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 541, constant_args_idx: 753, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2357, DY: %view_2320, DBETA: %full_default_152, DGAMMA: %as_strided_default_135, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_129 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 542, constant_args_idx: 754, grid: [(1024, 98, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2357, DY: %view_2320, INVSTD: %rsqrt_26, GAMMA: %primals_160, DBETA: %full_default_152, DGAMMA: %as_strided_default_135, DX: %permute_183, M: 100352, HW: 196, stride_n: 200704, stride_c: 196, BLOCK_M: 1024}})
triton_poi_fused_add_mul_37 = async_compile.triton('triton_poi_fused_add_mul_37', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1849688064}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/3z/c3zwsuyflcygt7uysropxi4zbwgoecyxtskmpy5cyvvg4ho37osz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_67, clone_default_68, clone_default_69
# Graph fragment:
#   %clone_default_68 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_136,), kwargs = {})
#   %clone_default_69 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_138,), kwargs = {})
#   %clone_default_67 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_134,), kwargs = {})
triton_poi_fused_38 = async_compile.triton('triton_poi_fused_38', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 28672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_38(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/bn/cbnthyprz7kqq7bx37v6igdusangy5pdqrbyb7pjxcvkmtgdn6wq.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_119, triton_kernel_wrapper_mutation_120
# Graph fragment:
#   %triton_kernel_wrapper_mutation_120 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 549, constant_args_idx: 763, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2435, DY: %view_2433, DBETA: %as_strided_default_125, DGAMMA: %as_strided_default_127, M: 401408, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_119 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 550, constant_args_idx: 764, grid: [(256, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2435, DY: %view_2433, INVSTD: %rsqrt_24, GAMMA: %primals_148, DBETA: %as_strided_default_125, DGAMMA: %as_strided_default_127, DX: %permute_185, M: 401408, HW: 784, stride_n: 200704, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_39 = async_compile.triton('triton_poi_fused_39', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_39(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/n5/cn5wdka6oopw5sybz23fu7mcm247pcwtchyhoxifkmwx35qjyfjo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_207 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1027, torch.float32), kwargs = {})
triton_poi_fused__to_copy_40 = async_compile.triton('triton_poi_fused__to_copy_40', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1310720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_40(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/q3/cq3hfc2bg2q2shmib6jb4haijydyeaeuxldgkl5xdtlslsftdq7k.py
# Topologically Sorted Source Nodes: [full_342], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_342 => full_default_395
# Graph fragment:
#   %full_default_395 : [num_users=6] = call_function[target=torch.ops.aten.full.default](args = ([401408, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_41 = async_compile.triton('triton_poi_fused_zeros_41', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_41(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5f/c5fp3icxxx6if2dimptlgdepuh4lsxwovqbpzmb6moh4pg6g7oeo.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_61
# Graph fragment:
#   %clone_default_61 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_122,), kwargs = {})
triton_poi_fused_42 = async_compile.triton('triton_poi_fused_42', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 616562688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/tn/ctnv6wqwd2u7a7kppbd6mf2te6ki7nz3gr3bjb7hwq7ocp5u6fwz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_59, clone_default_60
# Graph fragment:
#   %clone_default_59 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_118,), kwargs = {})
#   %clone_default_60 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_120,), kwargs = {})
triton_poi_fused_43 = async_compile.triton('triton_poi_fused_43', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_43(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2d/c2dm6hk4haxeiaryujedwu2qxu7ju72nl7hupgio6r6tfcqcmufz.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_114, triton_kernel_wrapper_mutation_115
# Graph fragment:
#   %triton_kernel_wrapper_mutation_115 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 553, constant_args_idx: 768, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2474, DY: %view_2472, DBETA: %as_strided_default_119, DGAMMA: %as_strided_default_121, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_114 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 554, constant_args_idx: 769, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2474, DY: %view_2472, INVSTD: %rsqrt_23, GAMMA: %primals_142, DBETA: %as_strided_default_119, DGAMMA: %as_strided_default_121, DX: %permute_186, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_44 = async_compile.triton('triton_poi_fused_44', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_44(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/wx/cwx4ifekiuv3idii4wgna4mt43pfydznil6n7kr5p5mcxmldz4kc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_212 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1039, torch.float32), kwargs = {})
triton_poi_fused__to_copy_45 = async_compile.triton('triton_poi_fused__to_copy_45', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 655360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sg/csglesz6wsr7ve6e4wmbxwz3sv5dh6u6xy2ny77et4wavjndhfb2.py
# Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   layer2_0_bn1 => full_default_64
# Graph fragment:
#   %full_default_64 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_46 = async_compile.triton('triton_poi_fused_zeros_46', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_46(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/it/citnejehtuldffotfqpssdp2rrob4s5gntfcsifqwa4m4r6pxdfx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_53, clone_default_54, clone_default_56, clone_default_57
# Graph fragment:
#   %clone_default_56 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_112,), kwargs = {})
#   %clone_default_57 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_114,), kwargs = {})
#   %clone_default_53 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_106,), kwargs = {})
#   %clone_default_54 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_108,), kwargs = {})
triton_poi_fused_47 = async_compile.triton('triton_poi_fused_47', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_47(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/sq/csq6rp2jk24b6rymlo2bdzqjuts26tzzw64jibcxkzkwxwghfnog.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_109, triton_kernel_wrapper_mutation_110
# Graph fragment:
#   %triton_kernel_wrapper_mutation_110 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 557, constant_args_idx: 773, grid: [(128, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2513, DY: %view_2511, DBETA: %as_strided_default_113, DGAMMA: %as_strided_default_115, M: 401408, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_109 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 558, constant_args_idx: 774, grid: [(128, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2513, DY: %view_2511, INVSTD: %rsqrt_22, GAMMA: %primals_136, DBETA: %as_strided_default_113, DGAMMA: %as_strided_default_115, DX: %permute_187, M: 401408, HW: 784, stride_n: 100352, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_48 = async_compile.triton('triton_poi_fused_48', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 102760448, 'x': 411041792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_48(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/7h/c7htxkuyhgxuctvomfpqlnbqwyl3qbhgijwxmabjq7vlvd4pp6bj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_217 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1051, torch.float32), kwargs = {})
triton_poi_fused__to_copy_49 = async_compile.triton('triton_poi_fused__to_copy_49', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1474560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xi/cxiv4ukx4zvzv2t47d75qdjnjcilkidwnclkz47ljurlbt563o3g.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_43, clone_default_52
# Graph fragment:
#   %clone_default_52 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_104,), kwargs = {})
#   %clone_default_43 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_86,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i8', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1027604480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_50(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3w/c3w66cilgbwq2lggzkcuqaorvcnkacqli3gvizcv7gkv5yiq472m.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_236 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_988, %getitem_1023), kwargs = {})
#   %mul_398 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_236, %view_2457), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_398, %getitem_1059), kwargs = {})
#   %mul_401 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_237, %view_2574), kwargs = {})
triton_poi_fused_add_mul_51 = async_compile.triton('triton_poi_fused_add_mul_51', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1233125376, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/43/c43fs73p45py6dwo7pgtg5j67boji6ejpwmlrehchll36mdpfxma.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_84, triton_kernel_wrapper_mutation_85
# Graph fragment:
#   %triton_kernel_wrapper_mutation_85 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 577, constant_args_idx: 798, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2708, DY: %view_2706, DBETA: %as_strided_default_83, DGAMMA: %as_strided_default_85, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_84 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 578, constant_args_idx: 799, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2708, DY: %view_2706, INVSTD: %rsqrt_17, GAMMA: %primals_106, DBETA: %as_strided_default_83, DGAMMA: %as_strided_default_85, DX: %permute_192, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_52 = async_compile.triton('triton_poi_fused_52', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 2055208960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_52(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/zc/czcb4zwnq4p65p5rdkgvq3tshpqkvfdbzxzn35f6ayz7nvs74aa2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_65, triton_kernel_wrapper_mutation_66, triton_kernel_wrapper_mutation_69, triton_kernel_wrapper_mutation_70
# Graph fragment:
#   %add_238 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_401, %getitem_1095), kwargs = {})
#   %mul_404 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_238, %view_2691), kwargs = {})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_404, %getitem_1131), kwargs = {})
#   %mul_407 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_239, %view_2808), kwargs = {})
#   %triton_kernel_wrapper_mutation_70 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 589, constant_args_idx: 813, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2825, DY: %view_2823, DBETA: %as_strided_default_65, DGAMMA: %as_strided_default_67, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_69 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 590, constant_args_idx: 814, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2825, DY: %view_2823, INVSTD: %rsqrt_14, GAMMA: %primals_88, DBETA: %as_strided_default_65, DGAMMA: %as_strided_default_67, DX: %permute_195, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_66 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 593, constant_args_idx: 817, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2860, DY: %view_2823, DBETA: %full_default_76, DGAMMA: %as_strided_default_63, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_65 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 594, constant_args_idx: 818, grid: [(512, 392, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2860, DY: %view_2823, INVSTD: %rsqrt_13, GAMMA: %primals_82, DBETA: %full_default_76, DGAMMA: %as_strided_default_63, DX: %permute_196, M: 401408, HW: 784, stride_n: 401408, stride_c: 784, BLOCK_M: 1024}})
triton_poi_fused_add_mul_53 = async_compile.triton('triton_poi_fused_add_mul_53', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 3699376128}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/qe/cqeftwvu3kk3bwukdxokare4e7s7dlsdzxttuk4nu4i6xosqeq7r.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_31, clone_default_32, clone_default_33
# Graph fragment:
#   %clone_default_32 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_64,), kwargs = {})
#   %clone_default_33 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_66,), kwargs = {})
#   %clone_default_31 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_62,), kwargs = {})
triton_poi_fused_54 = async_compile.triton('triton_poi_fused_54', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_54(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/5e/c5e4haqsa2zbfrij4kbtsrjodi3254gqccbyuunoauwwuauchts2.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_28, clone_default_29, clone_default_30
# Graph fragment:
#   %clone_default_29 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_58,), kwargs = {})
#   %clone_default_30 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_60,), kwargs = {})
#   %clone_default_28 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_56,), kwargs = {})
triton_poi_fused_55 = async_compile.triton('triton_poi_fused_55', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_55(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/dy/cdycrzo2wrwlgvqmnjhlrsmqp7qzqmkk3d3kdjwlye5lfexhm225.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_55, triton_kernel_wrapper_mutation_56
# Graph fragment:
#   %triton_kernel_wrapper_mutation_56 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 601, constant_args_idx: 827, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2938, DY: %view_2936, DBETA: %full_default_64, DGAMMA: %as_strided_default_57, M: 1605632, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_55 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 602, constant_args_idx: 828, grid: [(128, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2938, DY: %view_2936, INVSTD: %rsqrt_11, GAMMA: %primals_70, DBETA: %full_default_64, DGAMMA: %as_strided_default_57, DX: %permute_198, M: 1605632, HW: 3136, stride_n: 401408, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_56 = async_compile.triton('triton_poi_fused_56', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 411041792, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_56(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/fg/cfg5gom5lgtzm3l7gpbt6zxmbosl2tipmolznak4a53vgov6t5rd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_272 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1182, torch.float32), kwargs = {})
triton_poi_fused__to_copy_57 = async_compile.triton('triton_poi_fused__to_copy_57', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 327680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_57(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ia/ciaaa66g4a4sazrdvjwwuwfekgw3alngdfyagfzvyijzhjt5riqe.py
# Topologically Sorted Source Nodes: [full_380], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_380 => full_default_433
# Graph fragment:
#   %full_default_433 : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([802816, 512], 0), kwargs = {dtype: torch.int8, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_58 = async_compile.triton('triton_poi_fused_zeros_58', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_58(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int8)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/we/cwenkgqolmojes3cmum7us2yvikav6bdaikih57fi6imu7qslbqh.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_27
# Graph fragment:
#   %clone_default_27 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_54,), kwargs = {})
triton_poi_fused_59 = async_compile.triton('triton_poi_fused_59', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1233125376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_59(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xq/cxq2pkajyujrsllkf3vk2bdhogr6bcqxf3vswewe34o6z4o4ry2w.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_50, triton_kernel_wrapper_mutation_51
# Graph fragment:
#   %triton_kernel_wrapper_mutation_51 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 605, constant_args_idx: 832, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2977, DY: %view_2975, DBETA: %as_strided_default_51, DGAMMA: %as_strided_default_53, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_50 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 606, constant_args_idx: 833, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_2977, DY: %view_2975, INVSTD: %rsqrt_10, GAMMA: %primals_64, DBETA: %as_strided_default_51, DGAMMA: %as_strided_default_53, DX: %permute_199, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_60 = async_compile.triton('triton_poi_fused_60', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1644167168, 'x': 3288334336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_60(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/36/c36k55fxx2xtiy7zbqnm5nzhdmz7rcggesyg2fxwgvpgioy5h6xv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_277 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1194, torch.float32), kwargs = {})
triton_poi_fused__to_copy_61 = async_compile.triton('triton_poi_fused__to_copy_61', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 163840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_61(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2d/c2dlxgrv7x3ip5awoybvd3qc46eekfzfumqnokkpba73xi2xqwfp.py
# Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   bn1 => full_default
# Graph fragment:
#   %full_default : [num_users=16] = call_function[target=torch.ops.aten.full.default](args = ([64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_62 = async_compile.triton('triton_poi_fused_zeros_62', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_62(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wm/cwmap6ovlzqyofscr57u2w4plvfvmsbnj7jrgbidnq5nmkkdnoq5.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_19, clone_default_20, clone_default_22, clone_default_23
# Graph fragment:
#   %clone_default_22 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_44,), kwargs = {})
#   %clone_default_23 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_46,), kwargs = {})
#   %clone_default_19 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_38,), kwargs = {})
#   %clone_default_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_40,), kwargs = {})
triton_poi_fused_63 = async_compile.triton('triton_poi_fused_63', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_63(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/tc/ctcoyjvcppxbieeqvmgjv3avab44anipysbpw74c3wh6dufhpdxo.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_45, triton_kernel_wrapper_mutation_46
# Graph fragment:
#   %triton_kernel_wrapper_mutation_46 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 609, constant_args_idx: 837, grid: [(64, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3016, DY: %view_3014, DBETA: %as_strided_default_45, DGAMMA: %as_strided_default_47, M: 1605632, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_45 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 610, constant_args_idx: 838, grid: [(64, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3016, DY: %view_3014, INVSTD: %rsqrt_9, GAMMA: %primals_58, DBETA: %as_strided_default_45, DGAMMA: %as_strided_default_47, DX: %permute_200, M: 1605632, HW: 3136, stride_n: 200704, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_64 = async_compile.triton('triton_poi_fused_64', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 205520896, 'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_64(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/zf/czfbqkonzbwmpysabqqzikto6usu7cvmm7sw6vqqyvkbrvmcvr7y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_282 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1206, torch.float32), kwargs = {})
triton_poi_fused__to_copy_65 = async_compile.triton('triton_poi_fused__to_copy_65', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 368640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_65(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xx/cxxdtvj3lpiq6r6sxwdqzatyiket5fyybcfo7j4klxkiue4dx4bp.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_18, clone_default_9
# Graph fragment:
#   %clone_default_18 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_36,), kwargs = {})
#   %clone_default_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_18,), kwargs = {})
triton_poi_fused_66 = async_compile.triton('triton_poi_fused_66', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2055208960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_66(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fs/cfskuxpy6rd57bcxsbxrwobpwyq3ltijs5zjdkwfcgzhcge3c4zj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_240 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1143, %getitem_1178), kwargs = {})
#   %mul_410 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_240, %view_2960), kwargs = {})
#   %add_241 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_410, %getitem_1214), kwargs = {})
#   %mul_413 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_241, %view_3077), kwargs = {})
triton_poi_fused_add_mul_67 = async_compile.triton('triton_poi_fused_add_mul_67', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2466250752, 'x': 1644167168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/js/cjs4u7txsw3j7eknbqd7fk6ks5flpqzafz7zhaf625ca6enm6zdw.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default_6, clone_default_7, clone_default_8
# Graph fragment:
#   %clone_default_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_14,), kwargs = {})
#   %clone_default_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_16,), kwargs = {})
#   %clone_default_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_12,), kwargs = {})
triton_poi_fused_68 = async_compile.triton('triton_poi_fused_68', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_68(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/gr/cgr7btbefqundrxkr7ynw2uy4jpvss4gt4qk75e2lpanjxvkuvzq.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_16, triton_kernel_wrapper_mutation_17, triton_kernel_wrapper_mutation_20, triton_kernel_wrapper_mutation_21
# Graph fragment:
#   %triton_kernel_wrapper_mutation_21 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 629, constant_args_idx: 862, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3211, DY: %view_3209, DBETA: %as_strided_default_15, DGAMMA: %as_strided_default_17, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_20 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 630, constant_args_idx: 863, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3211, DY: %view_3209, INVSTD: %rsqrt_4, GAMMA: %primals_28, DBETA: %as_strided_default_15, DGAMMA: %as_strided_default_17, DX: %permute_205, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_17 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 633, constant_args_idx: 866, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3246, DY: %view_3209, DBETA: %full_default_18, DGAMMA: %as_strided_default_13, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_16 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 634, constant_args_idx: 867, grid: [(256, 1568, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3246, DY: %view_3209, INVSTD: %rsqrt_3, GAMMA: %primals_22, DBETA: %full_default_18, DGAMMA: %as_strided_default_13, DX: %permute_206, M: 1605632, HW: 3136, stride_n: 802816, stride_c: 3136, BLOCK_M: 1024}})
triton_poi_fused_69 = async_compile.triton('triton_poi_fused_69', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 822083584, 'x': 7398752256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_69(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/nd/cndcvgtfzrmj6yjrzuexymzfno3mlv5f4xx64akyrgq2z47rb3d6.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => clone_default, clone_default_1, clone_default_2, clone_default_3, clone_default_4
# Graph fragment:
#   %clone_default_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_6,), kwargs = {})
#   %clone_default_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_8,), kwargs = {})
#   %clone_default_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_2,), kwargs = {})
#   %clone_default_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default_4,), kwargs = {})
#   %clone_default : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%as_strided_default,), kwargs = {})
triton_poi_fused_70 = async_compile.triton('triton_poi_fused_70', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_70(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/cq/ccqzfuvzoicwpo3ioceik5l7uao65q3vd4ztpx4m2tliex77gqww.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_322 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1301, torch.float32), kwargs = {})
triton_poi_fused__to_copy_71 = async_compile.triton('triton_poi_fused__to_copy_71', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_71', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 40960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_71(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sq/csqvnaevrxkr6mza3xtrvm7utjidm4oqauhwbnfjxfg25iugvr2p.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1262, %getitem_1297), kwargs = {})
triton_poi_fused_add_72 = async_compile.triton('triton_poi_fused_add_72', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_72', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_72(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/k3/ck3qwxbcgzjurj6ybgn7cbjmahp5tvrcq3eas2yzy2zqdeiatapo.py
# Topologically Sorted Source Nodes: [maxpool], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
# Source node to ATen node mapping:
#   maxpool => _low_memory_max_pool_offsets_to_indices
# Graph fragment:
#   %add_243 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1262, %getitem_1297), kwargs = {})
#   %_low_memory_max_pool_offsets_to_indices : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_offsets_to_indices.default](args = (%getitem_14, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]), kwargs = {})
#   %max_pool2d_with_indices_backward : [num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%add_243, %getitem_10, [3, 3], [2, 2], [1, 1], [1, 1], False, %_low_memory_max_pool_offsets_to_indices), kwargs = {})
triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_73 = async_compile.triton('triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_73', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_73', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_73(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/kf/ckf26otjbkkwp2azjoooyfowegcbynbfgxcmzlzr4qiefquo2g5q.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_1, triton_kernel_wrapper_mutation_2
# Graph fragment:
#   %triton_kernel_wrapper_mutation_2 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 645, constant_args_idx: 881, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3363, DY: %view_3361, DBETA: %full_default, DGAMMA: %as_strided_default_1, M: 6422528, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
#   %triton_kernel_wrapper_mutation_1 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 646, constant_args_idx: 882, grid: [(64, 6272, 1)], tma_descriptor_metadata: {}, kwargs: {X_hat: %view_3363, DY: %view_3361, INVSTD: %rsqrt, GAMMA: %primals_4, DBETA: %full_default, DGAMMA: %as_strided_default_1, DX: %permute_209, M: 6422528, HW: 12544, stride_n: 802816, stride_c: 12544, BLOCK_M: 1024}})
triton_poi_fused_74 = async_compile.triton('triton_poi_fused_74', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_74', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4110417920}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_74(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/x6/cx6aaecswfeodfvamhjgeb645sijbdlqrpd4kuyfobdtjb25ih52.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_327 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1310, torch.float32), kwargs = {})
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_75', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 94080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_75(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    assert_size_stride(getitem, (150528, 32), (32, 1))
    assert_size_stride(getitem_1, (150528, ), (1, ))
    assert_size_stride(getitem_2, (150528, ), (1, ))
    assert_size_stride(rsqrt, (64, ), (1, ))
    assert_size_stride(getitem_7, (802816, 32), (32, 1))
    assert_size_stride(getitem_8, (802816, ), (1, ))
    assert_size_stride(getitem_9, (802816, ), (1, ))
    assert_size_stride(getitem_10, (512, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(getitem_12, (802816, 16), (16, 1))
    assert_size_stride(getitem_14, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_2, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_15, (200704, 32), (32, 1))
    assert_size_stride(getitem_16, (200704, ), (1, ))
    assert_size_stride(getitem_17, (200704, ), (1, ))
    assert_size_stride(rsqrt_1, (64, ), (1, ))
    assert_size_stride(getitem_22, (200704, 32), (32, 1))
    assert_size_stride(getitem_23, (200704, ), (1, ))
    assert_size_stride(getitem_24, (200704, ), (1, ))
    assert_size_stride(getitem_27, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_3, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(getitem_28, (200704, 32), (32, 1))
    assert_size_stride(getitem_29, (200704, ), (1, ))
    assert_size_stride(getitem_30, (200704, ), (1, ))
    assert_size_stride(rsqrt_2, (64, ), (1, ))
    assert_size_stride(getitem_35, (200704, 32), (32, 1))
    assert_size_stride(getitem_36, (200704, ), (1, ))
    assert_size_stride(getitem_37, (200704, ), (1, ))
    assert_size_stride(getitem_40, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_4, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_41, (200704, 32), (32, 1))
    assert_size_stride(getitem_42, (200704, ), (1, ))
    assert_size_stride(getitem_43, (200704, ), (1, ))
    assert_size_stride(rsqrt_3, (256, ), (1, ))
    assert_size_stride(getitem_48, (802816, 32), (32, 1))
    assert_size_stride(getitem_49, (802816, ), (1, ))
    assert_size_stride(getitem_50, (802816, ), (1, ))
    assert_size_stride(convert_element_type_5, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_51, (200704, 32), (32, 1))
    assert_size_stride(getitem_52, (200704, ), (1, ))
    assert_size_stride(getitem_53, (200704, ), (1, ))
    assert_size_stride(rsqrt_4, (256, ), (1, ))
    assert_size_stride(getitem_58, (802816, 32), (32, 1))
    assert_size_stride(getitem_59, (802816, ), (1, ))
    assert_size_stride(getitem_60, (802816, ), (1, ))
    assert_size_stride(getitem_63, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_6, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_64, (802816, 32), (32, 1))
    assert_size_stride(getitem_65, (802816, ), (1, ))
    assert_size_stride(getitem_66, (802816, ), (1, ))
    assert_size_stride(rsqrt_5, (64, ), (1, ))
    assert_size_stride(getitem_71, (200704, 32), (32, 1))
    assert_size_stride(getitem_72, (200704, ), (1, ))
    assert_size_stride(getitem_73, (200704, ), (1, ))
    assert_size_stride(getitem_76, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_7, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(getitem_77, (200704, 32), (32, 1))
    assert_size_stride(getitem_78, (200704, ), (1, ))
    assert_size_stride(getitem_79, (200704, ), (1, ))
    assert_size_stride(rsqrt_6, (64, ), (1, ))
    assert_size_stride(getitem_84, (200704, 32), (32, 1))
    assert_size_stride(getitem_85, (200704, ), (1, ))
    assert_size_stride(getitem_86, (200704, ), (1, ))
    assert_size_stride(getitem_89, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_8, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_90, (200704, 32), (32, 1))
    assert_size_stride(getitem_91, (200704, ), (1, ))
    assert_size_stride(getitem_92, (200704, ), (1, ))
    assert_size_stride(rsqrt_7, (256, ), (1, ))
    assert_size_stride(getitem_97, (802816, 32), (32, 1))
    assert_size_stride(getitem_98, (802816, ), (1, ))
    assert_size_stride(getitem_99, (802816, ), (1, ))
    assert_size_stride(getitem_102, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_9, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_103, (802816, 32), (32, 1))
    assert_size_stride(getitem_104, (802816, ), (1, ))
    assert_size_stride(getitem_105, (802816, ), (1, ))
    assert_size_stride(rsqrt_8, (64, ), (1, ))
    assert_size_stride(getitem_110, (200704, 32), (32, 1))
    assert_size_stride(getitem_111, (200704, ), (1, ))
    assert_size_stride(getitem_112, (200704, ), (1, ))
    assert_size_stride(getitem_115, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_10, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(getitem_116, (200704, 32), (32, 1))
    assert_size_stride(getitem_117, (200704, ), (1, ))
    assert_size_stride(getitem_118, (200704, ), (1, ))
    assert_size_stride(rsqrt_9, (64, ), (1, ))
    assert_size_stride(getitem_123, (200704, 32), (32, 1))
    assert_size_stride(getitem_124, (200704, ), (1, ))
    assert_size_stride(getitem_125, (200704, ), (1, ))
    assert_size_stride(getitem_128, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_11, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(getitem_129, (200704, 32), (32, 1))
    assert_size_stride(getitem_130, (200704, ), (1, ))
    assert_size_stride(getitem_131, (200704, ), (1, ))
    assert_size_stride(rsqrt_10, (256, ), (1, ))
    assert_size_stride(getitem_136, (802816, 32), (32, 1))
    assert_size_stride(getitem_137, (802816, ), (1, ))
    assert_size_stride(getitem_138, (802816, ), (1, ))
    assert_size_stride(getitem_141, (802816, 16), (16, 1))
    assert_size_stride(convert_element_type_12, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_142, (802816, 32), (32, 1))
    assert_size_stride(getitem_143, (802816, ), (1, ))
    assert_size_stride(getitem_144, (802816, ), (1, ))
    assert_size_stride(rsqrt_11, (128, ), (1, ))
    assert_size_stride(getitem_149, (401408, 32), (32, 1))
    assert_size_stride(getitem_150, (401408, ), (1, ))
    assert_size_stride(getitem_151, (401408, ), (1, ))
    assert_size_stride(getitem_154, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_13, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(getitem_155, (401408, 32), (32, 1))
    assert_size_stride(getitem_156, (401408, ), (1, ))
    assert_size_stride(getitem_157, (401408, ), (1, ))
    assert_size_stride(rsqrt_12, (128, ), (1, ))
    assert_size_stride(getitem_162, (100352, 32), (32, 1))
    assert_size_stride(getitem_163, (100352, ), (1, ))
    assert_size_stride(getitem_164, (100352, ), (1, ))
    assert_size_stride(getitem_167, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_14, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(getitem_168, (100352, 32), (32, 1))
    assert_size_stride(getitem_169, (100352, ), (1, ))
    assert_size_stride(getitem_170, (100352, ), (1, ))
    assert_size_stride(rsqrt_13, (512, ), (1, ))
    assert_size_stride(getitem_175, (401408, 32), (32, 1))
    assert_size_stride(getitem_176, (401408, ), (1, ))
    assert_size_stride(getitem_177, (401408, ), (1, ))
    assert_size_stride(convert_element_type_15, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_178, (802816, 32), (32, 1))
    assert_size_stride(getitem_179, (802816, ), (1, ))
    assert_size_stride(getitem_180, (802816, ), (1, ))
    assert_size_stride(rsqrt_14, (512, ), (1, ))
    assert_size_stride(getitem_185, (401408, 32), (32, 1))
    assert_size_stride(getitem_186, (401408, ), (1, ))
    assert_size_stride(getitem_187, (401408, ), (1, ))
    assert_size_stride(getitem_190, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_16, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_191, (401408, 32), (32, 1))
    assert_size_stride(getitem_192, (401408, ), (1, ))
    assert_size_stride(getitem_193, (401408, ), (1, ))
    assert_size_stride(rsqrt_15, (128, ), (1, ))
    assert_size_stride(getitem_198, (100352, 32), (32, 1))
    assert_size_stride(getitem_199, (100352, ), (1, ))
    assert_size_stride(getitem_200, (100352, ), (1, ))
    assert_size_stride(getitem_203, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_17, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(getitem_204, (100352, 32), (32, 1))
    assert_size_stride(getitem_205, (100352, ), (1, ))
    assert_size_stride(getitem_206, (100352, ), (1, ))
    assert_size_stride(rsqrt_16, (128, ), (1, ))
    assert_size_stride(getitem_211, (100352, 32), (32, 1))
    assert_size_stride(getitem_212, (100352, ), (1, ))
    assert_size_stride(getitem_213, (100352, ), (1, ))
    assert_size_stride(getitem_216, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_18, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(getitem_217, (100352, 32), (32, 1))
    assert_size_stride(getitem_218, (100352, ), (1, ))
    assert_size_stride(getitem_219, (100352, ), (1, ))
    assert_size_stride(rsqrt_17, (512, ), (1, ))
    assert_size_stride(getitem_224, (401408, 32), (32, 1))
    assert_size_stride(getitem_225, (401408, ), (1, ))
    assert_size_stride(getitem_226, (401408, ), (1, ))
    assert_size_stride(getitem_229, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_19, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_230, (401408, 32), (32, 1))
    assert_size_stride(getitem_231, (401408, ), (1, ))
    assert_size_stride(getitem_232, (401408, ), (1, ))
    assert_size_stride(rsqrt_18, (128, ), (1, ))
    assert_size_stride(getitem_237, (100352, 32), (32, 1))
    assert_size_stride(getitem_238, (100352, ), (1, ))
    assert_size_stride(getitem_239, (100352, ), (1, ))
    assert_size_stride(getitem_242, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_20, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(getitem_243, (100352, 32), (32, 1))
    assert_size_stride(getitem_244, (100352, ), (1, ))
    assert_size_stride(getitem_245, (100352, ), (1, ))
    assert_size_stride(rsqrt_19, (128, ), (1, ))
    assert_size_stride(getitem_250, (100352, 32), (32, 1))
    assert_size_stride(getitem_251, (100352, ), (1, ))
    assert_size_stride(getitem_252, (100352, ), (1, ))
    assert_size_stride(getitem_255, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_21, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(getitem_256, (100352, 32), (32, 1))
    assert_size_stride(getitem_257, (100352, ), (1, ))
    assert_size_stride(getitem_258, (100352, ), (1, ))
    assert_size_stride(rsqrt_20, (512, ), (1, ))
    assert_size_stride(getitem_263, (401408, 32), (32, 1))
    assert_size_stride(getitem_264, (401408, ), (1, ))
    assert_size_stride(getitem_265, (401408, ), (1, ))
    assert_size_stride(getitem_268, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_22, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_269, (401408, 32), (32, 1))
    assert_size_stride(getitem_270, (401408, ), (1, ))
    assert_size_stride(getitem_271, (401408, ), (1, ))
    assert_size_stride(rsqrt_21, (128, ), (1, ))
    assert_size_stride(getitem_276, (100352, 32), (32, 1))
    assert_size_stride(getitem_277, (100352, ), (1, ))
    assert_size_stride(getitem_278, (100352, ), (1, ))
    assert_size_stride(getitem_281, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_23, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(getitem_282, (100352, 32), (32, 1))
    assert_size_stride(getitem_283, (100352, ), (1, ))
    assert_size_stride(getitem_284, (100352, ), (1, ))
    assert_size_stride(rsqrt_22, (128, ), (1, ))
    assert_size_stride(getitem_289, (100352, 32), (32, 1))
    assert_size_stride(getitem_290, (100352, ), (1, ))
    assert_size_stride(getitem_291, (100352, ), (1, ))
    assert_size_stride(getitem_294, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_24, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(getitem_295, (100352, 32), (32, 1))
    assert_size_stride(getitem_296, (100352, ), (1, ))
    assert_size_stride(getitem_297, (100352, ), (1, ))
    assert_size_stride(rsqrt_23, (512, ), (1, ))
    assert_size_stride(getitem_302, (401408, 32), (32, 1))
    assert_size_stride(getitem_303, (401408, ), (1, ))
    assert_size_stride(getitem_304, (401408, ), (1, ))
    assert_size_stride(getitem_307, (401408, 16), (16, 1))
    assert_size_stride(convert_element_type_25, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_308, (401408, 32), (32, 1))
    assert_size_stride(getitem_309, (401408, ), (1, ))
    assert_size_stride(getitem_310, (401408, ), (1, ))
    assert_size_stride(rsqrt_24, (256, ), (1, ))
    assert_size_stride(getitem_315, (200704, 32), (32, 1))
    assert_size_stride(getitem_316, (200704, ), (1, ))
    assert_size_stride(getitem_317, (200704, ), (1, ))
    assert_size_stride(getitem_320, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_26, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_321, (200704, 32), (32, 1))
    assert_size_stride(getitem_322, (200704, ), (1, ))
    assert_size_stride(getitem_323, (200704, ), (1, ))
    assert_size_stride(rsqrt_25, (256, ), (1, ))
    assert_size_stride(getitem_328, (50176, 32), (32, 1))
    assert_size_stride(getitem_329, (50176, ), (1, ))
    assert_size_stride(getitem_330, (50176, ), (1, ))
    assert_size_stride(getitem_333, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_27, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_334, (50176, 32), (32, 1))
    assert_size_stride(getitem_335, (50176, ), (1, ))
    assert_size_stride(getitem_336, (50176, ), (1, ))
    assert_size_stride(rsqrt_26, (1024, ), (1, ))
    assert_size_stride(getitem_341, (200704, 32), (32, 1))
    assert_size_stride(getitem_342, (200704, ), (1, ))
    assert_size_stride(getitem_343, (200704, ), (1, ))
    assert_size_stride(convert_element_type_28, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_344, (401408, 32), (32, 1))
    assert_size_stride(getitem_345, (401408, ), (1, ))
    assert_size_stride(getitem_346, (401408, ), (1, ))
    assert_size_stride(rsqrt_27, (1024, ), (1, ))
    assert_size_stride(getitem_351, (200704, 32), (32, 1))
    assert_size_stride(getitem_352, (200704, ), (1, ))
    assert_size_stride(getitem_353, (200704, ), (1, ))
    assert_size_stride(getitem_356, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_29, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_357, (200704, 32), (32, 1))
    assert_size_stride(getitem_358, (200704, ), (1, ))
    assert_size_stride(getitem_359, (200704, ), (1, ))
    assert_size_stride(rsqrt_28, (256, ), (1, ))
    assert_size_stride(getitem_364, (50176, 32), (32, 1))
    assert_size_stride(getitem_365, (50176, ), (1, ))
    assert_size_stride(getitem_366, (50176, ), (1, ))
    assert_size_stride(getitem_369, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_30, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_370, (50176, 32), (32, 1))
    assert_size_stride(getitem_371, (50176, ), (1, ))
    assert_size_stride(getitem_372, (50176, ), (1, ))
    assert_size_stride(rsqrt_29, (256, ), (1, ))
    assert_size_stride(getitem_377, (50176, 32), (32, 1))
    assert_size_stride(getitem_378, (50176, ), (1, ))
    assert_size_stride(getitem_379, (50176, ), (1, ))
    assert_size_stride(getitem_382, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_31, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_383, (50176, 32), (32, 1))
    assert_size_stride(getitem_384, (50176, ), (1, ))
    assert_size_stride(getitem_385, (50176, ), (1, ))
    assert_size_stride(rsqrt_30, (1024, ), (1, ))
    assert_size_stride(getitem_390, (200704, 32), (32, 1))
    assert_size_stride(getitem_391, (200704, ), (1, ))
    assert_size_stride(getitem_392, (200704, ), (1, ))
    assert_size_stride(getitem_395, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_32, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_396, (200704, 32), (32, 1))
    assert_size_stride(getitem_397, (200704, ), (1, ))
    assert_size_stride(getitem_398, (200704, ), (1, ))
    assert_size_stride(rsqrt_31, (256, ), (1, ))
    assert_size_stride(getitem_403, (50176, 32), (32, 1))
    assert_size_stride(getitem_404, (50176, ), (1, ))
    assert_size_stride(getitem_405, (50176, ), (1, ))
    assert_size_stride(getitem_408, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_33, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_409, (50176, 32), (32, 1))
    assert_size_stride(getitem_410, (50176, ), (1, ))
    assert_size_stride(getitem_411, (50176, ), (1, ))
    assert_size_stride(rsqrt_32, (256, ), (1, ))
    assert_size_stride(getitem_416, (50176, 32), (32, 1))
    assert_size_stride(getitem_417, (50176, ), (1, ))
    assert_size_stride(getitem_418, (50176, ), (1, ))
    assert_size_stride(getitem_421, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_34, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_422, (50176, 32), (32, 1))
    assert_size_stride(getitem_423, (50176, ), (1, ))
    assert_size_stride(getitem_424, (50176, ), (1, ))
    assert_size_stride(rsqrt_33, (1024, ), (1, ))
    assert_size_stride(getitem_429, (200704, 32), (32, 1))
    assert_size_stride(getitem_430, (200704, ), (1, ))
    assert_size_stride(getitem_431, (200704, ), (1, ))
    assert_size_stride(getitem_434, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_35, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_435, (200704, 32), (32, 1))
    assert_size_stride(getitem_436, (200704, ), (1, ))
    assert_size_stride(getitem_437, (200704, ), (1, ))
    assert_size_stride(rsqrt_34, (256, ), (1, ))
    assert_size_stride(getitem_442, (50176, 32), (32, 1))
    assert_size_stride(getitem_443, (50176, ), (1, ))
    assert_size_stride(getitem_444, (50176, ), (1, ))
    assert_size_stride(getitem_447, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_36, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_448, (50176, 32), (32, 1))
    assert_size_stride(getitem_449, (50176, ), (1, ))
    assert_size_stride(getitem_450, (50176, ), (1, ))
    assert_size_stride(rsqrt_35, (256, ), (1, ))
    assert_size_stride(getitem_455, (50176, 32), (32, 1))
    assert_size_stride(getitem_456, (50176, ), (1, ))
    assert_size_stride(getitem_457, (50176, ), (1, ))
    assert_size_stride(getitem_460, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_37, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_461, (50176, 32), (32, 1))
    assert_size_stride(getitem_462, (50176, ), (1, ))
    assert_size_stride(getitem_463, (50176, ), (1, ))
    assert_size_stride(rsqrt_36, (1024, ), (1, ))
    assert_size_stride(getitem_468, (200704, 32), (32, 1))
    assert_size_stride(getitem_469, (200704, ), (1, ))
    assert_size_stride(getitem_470, (200704, ), (1, ))
    assert_size_stride(getitem_473, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_38, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_474, (200704, 32), (32, 1))
    assert_size_stride(getitem_475, (200704, ), (1, ))
    assert_size_stride(getitem_476, (200704, ), (1, ))
    assert_size_stride(rsqrt_37, (256, ), (1, ))
    assert_size_stride(getitem_481, (50176, 32), (32, 1))
    assert_size_stride(getitem_482, (50176, ), (1, ))
    assert_size_stride(getitem_483, (50176, ), (1, ))
    assert_size_stride(getitem_486, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_39, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_487, (50176, 32), (32, 1))
    assert_size_stride(getitem_488, (50176, ), (1, ))
    assert_size_stride(getitem_489, (50176, ), (1, ))
    assert_size_stride(rsqrt_38, (256, ), (1, ))
    assert_size_stride(getitem_494, (50176, 32), (32, 1))
    assert_size_stride(getitem_495, (50176, ), (1, ))
    assert_size_stride(getitem_496, (50176, ), (1, ))
    assert_size_stride(getitem_499, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_40, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_500, (50176, 32), (32, 1))
    assert_size_stride(getitem_501, (50176, ), (1, ))
    assert_size_stride(getitem_502, (50176, ), (1, ))
    assert_size_stride(rsqrt_39, (1024, ), (1, ))
    assert_size_stride(getitem_507, (200704, 32), (32, 1))
    assert_size_stride(getitem_508, (200704, ), (1, ))
    assert_size_stride(getitem_509, (200704, ), (1, ))
    assert_size_stride(getitem_512, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_41, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_513, (200704, 32), (32, 1))
    assert_size_stride(getitem_514, (200704, ), (1, ))
    assert_size_stride(getitem_515, (200704, ), (1, ))
    assert_size_stride(rsqrt_40, (256, ), (1, ))
    assert_size_stride(getitem_520, (50176, 32), (32, 1))
    assert_size_stride(getitem_521, (50176, ), (1, ))
    assert_size_stride(getitem_522, (50176, ), (1, ))
    assert_size_stride(getitem_525, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_42, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(getitem_526, (50176, 32), (32, 1))
    assert_size_stride(getitem_527, (50176, ), (1, ))
    assert_size_stride(getitem_528, (50176, ), (1, ))
    assert_size_stride(rsqrt_41, (256, ), (1, ))
    assert_size_stride(getitem_533, (50176, 32), (32, 1))
    assert_size_stride(getitem_534, (50176, ), (1, ))
    assert_size_stride(getitem_535, (50176, ), (1, ))
    assert_size_stride(getitem_538, (50176, 16), (16, 1))
    assert_size_stride(convert_element_type_43, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(getitem_539, (50176, 32), (32, 1))
    assert_size_stride(getitem_540, (50176, ), (1, ))
    assert_size_stride(getitem_541, (50176, ), (1, ))
    assert_size_stride(rsqrt_42, (1024, ), (1, ))
    assert_size_stride(getitem_546, (200704, 32), (32, 1))
    assert_size_stride(getitem_547, (200704, ), (1, ))
    assert_size_stride(getitem_548, (200704, ), (1, ))
    assert_size_stride(getitem_551, (200704, 16), (16, 1))
    assert_size_stride(convert_element_type_44, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_552, (200704, 32), (32, 1))
    assert_size_stride(getitem_553, (200704, ), (1, ))
    assert_size_stride(getitem_554, (200704, ), (1, ))
    assert_size_stride(rsqrt_43, (512, ), (1, ))
    assert_size_stride(getitem_559, (100352, 32), (32, 1))
    assert_size_stride(getitem_560, (100352, ), (1, ))
    assert_size_stride(getitem_561, (100352, ), (1, ))
    assert_size_stride(getitem_564, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_45, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(getitem_565, (100352, 32), (32, 1))
    assert_size_stride(getitem_566, (100352, ), (1, ))
    assert_size_stride(getitem_567, (100352, ), (1, ))
    assert_size_stride(rsqrt_44, (512, ), (1, ))
    assert_size_stride(getitem_572, (25088, 32), (32, 1))
    assert_size_stride(getitem_573, (25088, ), (1, ))
    assert_size_stride(getitem_574, (25088, ), (1, ))
    assert_size_stride(getitem_577, (25088, 16), (16, 1))
    assert_size_stride(convert_element_type_46, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_578, (25088, 32), (32, 1))
    assert_size_stride(getitem_579, (25088, ), (1, ))
    assert_size_stride(getitem_580, (25088, ), (1, ))
    assert_size_stride(rsqrt_45, (2048, ), (1, ))
    assert_size_stride(getitem_585, (100352, 32), (32, 1))
    assert_size_stride(getitem_586, (100352, ), (1, ))
    assert_size_stride(getitem_587, (100352, ), (1, ))
    assert_size_stride(convert_element_type_47, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(getitem_588, (200704, 32), (32, 1))
    assert_size_stride(getitem_589, (200704, ), (1, ))
    assert_size_stride(getitem_590, (200704, ), (1, ))
    assert_size_stride(rsqrt_46, (2048, ), (1, ))
    assert_size_stride(getitem_595, (100352, 32), (32, 1))
    assert_size_stride(getitem_596, (100352, ), (1, ))
    assert_size_stride(getitem_597, (100352, ), (1, ))
    assert_size_stride(getitem_600, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_48, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(getitem_601, (100352, 32), (32, 1))
    assert_size_stride(getitem_602, (100352, ), (1, ))
    assert_size_stride(getitem_603, (100352, ), (1, ))
    assert_size_stride(rsqrt_47, (512, ), (1, ))
    assert_size_stride(getitem_608, (25088, 32), (32, 1))
    assert_size_stride(getitem_609, (25088, ), (1, ))
    assert_size_stride(getitem_610, (25088, ), (1, ))
    assert_size_stride(getitem_613, (25088, 16), (16, 1))
    assert_size_stride(convert_element_type_49, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(getitem_614, (25088, 32), (32, 1))
    assert_size_stride(getitem_615, (25088, ), (1, ))
    assert_size_stride(getitem_616, (25088, ), (1, ))
    assert_size_stride(rsqrt_48, (512, ), (1, ))
    assert_size_stride(getitem_621, (25088, 32), (32, 1))
    assert_size_stride(getitem_622, (25088, ), (1, ))
    assert_size_stride(getitem_623, (25088, ), (1, ))
    assert_size_stride(getitem_626, (25088, 16), (16, 1))
    assert_size_stride(convert_element_type_50, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_627, (25088, 32), (32, 1))
    assert_size_stride(getitem_628, (25088, ), (1, ))
    assert_size_stride(getitem_629, (25088, ), (1, ))
    assert_size_stride(rsqrt_49, (2048, ), (1, ))
    assert_size_stride(getitem_634, (100352, 32), (32, 1))
    assert_size_stride(getitem_635, (100352, ), (1, ))
    assert_size_stride(getitem_636, (100352, ), (1, ))
    assert_size_stride(getitem_639, (100352, 16), (16, 1))
    assert_size_stride(convert_element_type_51, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(getitem_640, (100352, 32), (32, 1))
    assert_size_stride(getitem_641, (100352, ), (1, ))
    assert_size_stride(getitem_642, (100352, ), (1, ))
    assert_size_stride(rsqrt_50, (512, ), (1, ))
    assert_size_stride(getitem_647, (25088, 32), (32, 1))
    assert_size_stride(getitem_648, (25088, ), (1, ))
    assert_size_stride(getitem_649, (25088, ), (1, ))
    assert_size_stride(getitem_652, (25088, 16), (16, 1))
    assert_size_stride(convert_element_type_52, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(getitem_653, (25088, 32), (32, 1))
    assert_size_stride(getitem_654, (25088, ), (1, ))
    assert_size_stride(getitem_655, (25088, ), (1, ))
    assert_size_stride(rsqrt_51, (512, ), (1, ))
    assert_size_stride(getitem_660, (25088, 32), (32, 1))
    assert_size_stride(getitem_661, (25088, ), (1, ))
    assert_size_stride(getitem_662, (25088, ), (1, ))
    assert_size_stride(getitem_665, (25088, 16), (16, 1))
    assert_size_stride(convert_element_type_53, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(getitem_666, (25088, 32), (32, 1))
    assert_size_stride(getitem_667, (25088, ), (1, ))
    assert_size_stride(getitem_668, (25088, ), (1, ))
    assert_size_stride(rsqrt_52, (2048, ), (1, ))
    assert_size_stride(getitem_673, (100352, 32), (32, 1))
    assert_size_stride(getitem_674, (100352, ), (1, ))
    assert_size_stride(getitem_675, (100352, ), (1, ))
    assert_size_stride(getitem_678, (100352, 16), (16, 1))
    assert_size_stride(getitem_679, (2048, 32), (32, 1))
    assert_size_stride(getitem_680, (2048, ), (1, ))
    assert_size_stride(getitem_681, (2048, ), (1, ))
    assert_size_stride(convert_element_type_54, (100, 2048), (2048, 1))
    assert_size_stride(tangents_1, (512, 100), (100, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_679, getitem_680, getitem_681, buf0, 32, 1, 512, 1, 2, 16, 32, 2048, 1, 1, stream=stream0)
        del getitem_679
        del getitem_680
        del getitem_681
        buf2 = empty_strided_cuda((104, 512), (1, 104), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_0.run(tangents_1, buf2, 53248, stream=stream0)
        buf3 = empty_strided_cuda((104, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf2, reinterpret_tensor(buf0, (512, 2048), (2048, 1), 0), out=buf3)
        del buf2
        buf4 = reinterpret_tensor(buf0, (512, 2048), (2048, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [view_1329], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, convert_element_type_54, out=buf4)
        del convert_element_type_54
        del tangents_1
        buf5 = empty_strided_cuda((100, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(buf3, buf5, 204800, stream=stream0)
        del buf3
        buf6 = empty_strided_cuda((100352, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_257], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_2.run(buf6, 51380224, stream=stream0)
        buf7 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf73 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf6, buf7, buf73, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_678, reinterpret_tensor(buf7, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_678
        buf9 = empty_strided_cuda((100352, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_673, getitem_674, getitem_675, buf9, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_673
        del getitem_674
        del getitem_675
        buf11 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer4_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_4.run(buf11, 2048, stream=stream0)
        buf12 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf13 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf77 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf78 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(buf11, buf12, buf13, buf77, buf78, 2048, stream=stream0)
        buf14 = empty_strided_cuda((512, 2048, 49), (100352, 49, 1), torch.bfloat16)
        buf18 = empty_strided_cuda((512, 2048, 49), (100352, 49, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(buf4, buf7, buf14, buf18, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf9, (512, 2048, 49), (100352, 49, 1), 0), buf14, buf12, buf13, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf17 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf9, (512, 2048, 49), (100352, 49, 1), 0), buf18, rsqrt_52, primals_316, buf12, buf13, buf17, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del primals_316
        del rsqrt_52
        buf20 = empty_strided_cuda((25088, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_666, getitem_667, getitem_668, buf20, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_666
        del getitem_667
        del getitem_668
        buf22 = empty_strided_cuda((1, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf23 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf17, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf22, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_53
        buf24 = buf23[0]
        assert_size_stride(buf24, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf24, 16, 'torch.ops.aten.convolution_backward.default')
        del buf23
        buf25 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf26 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf17, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf20, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf25, (2048, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf27 = buf26[1]
        assert_size_stride(buf27, (2048, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf27, 16, 'torch.ops.aten.convolution_backward.default')
        del buf26
        buf28 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf27, buf28, 1048576, stream=stream0)
        del buf27
        buf29 = empty_strided_cuda((25088, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_260], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_8.run(buf29, 12845056, stream=stream0)
        buf30 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        buf52 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(buf29, buf30, buf52, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_665, reinterpret_tensor(buf30, (25088, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        del getitem_665
        buf32 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_660, getitem_661, getitem_662, buf32, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_660
        del getitem_661
        del getitem_662
        buf34 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_10.run(buf34, 512, stream=stream0)
        buf35 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf36 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf56 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf57 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf34, buf35, buf36, buf56, buf57, 512, stream=stream0)
        buf37 = empty_strided_cuda((512, 512, 49), (25088, 49, 1), torch.bfloat16)
        buf41 = empty_strided_cuda((512, 512, 49), (25088, 49, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf24, buf30, buf37, buf41, 262144, 49, stream=stream0)
        del buf24
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf32, (512, 512, 49), (25088, 49, 1), 0), buf37, buf35, buf36, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf40 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf32, (512, 512, 49), (25088, 49, 1), 0), buf41, rsqrt_51, primals_310, buf35, buf36, buf40, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del primals_310
        del rsqrt_51
        buf43 = reinterpret_tensor(buf41, (25088, 512), (512, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_653, getitem_654, getitem_655, buf43, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_653
        del getitem_654
        del getitem_655
        buf45 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf46 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf40, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf45, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_52
        buf47 = buf46[0]
        assert_size_stride(buf47, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf47, 16, 'torch.ops.aten.convolution_backward.default')
        del buf46
        buf48 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf49 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf40, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf43, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf48, (512, 512, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf50 = buf49[1]
        assert_size_stride(buf50, (512, 512, 3, 3), (4608, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf50, 16, 'torch.ops.aten.convolution_backward.default')
        del buf49
        buf51 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf50, buf51, 2359296, stream=stream0)
        del buf50
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_652, reinterpret_tensor(buf52, (25088, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        del getitem_652
        buf54 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_647, getitem_648, getitem_649, buf54, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_647
        del getitem_648
        del getitem_649
        buf58 = buf40; del buf40  # reuse
        buf62 = reinterpret_tensor(buf32, (512, 512, 49), (25088, 49, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf47, buf52, buf58, buf62, 262144, 49, stream=stream0)
        del buf47
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf54, (512, 512, 49), (25088, 49, 1), 0), buf58, buf56, buf57, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf61 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf54, (512, 512, 49), (25088, 49, 1), 0), buf62, rsqrt_50, primals_304, buf56, buf57, buf61, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del primals_304
        del rsqrt_50
        buf64 = reinterpret_tensor(buf17, (100352, 512), (512, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_640, getitem_641, getitem_642, buf64, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_640
        del getitem_641
        del getitem_642
        buf66 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf67 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf61, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf66, (512, 2048, 7, 7), (0, 0, 0, 0), 0), convert_element_type_51, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_51
        buf68 = buf67[0]
        assert_size_stride(buf68, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf68, 16, 'torch.ops.aten.convolution_backward.default')
        del buf67
        buf69 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf70 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf61, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf64, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf69, (512, 2048, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf71 = buf70[1]
        assert_size_stride(buf71, (512, 2048, 1, 1), (2048, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf71, 16, 'torch.ops.aten.convolution_backward.default')
        del buf70
        buf72 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf71, buf72, 1048576, stream=stream0)
        del buf71
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_639, reinterpret_tensor(buf73, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_639
        buf75 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_634, getitem_635, getitem_636, buf75, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_634
        del getitem_635
        del getitem_636
        buf79 = reinterpret_tensor(buf9, (512, 2048, 49), (100352, 49, 1), 0); del buf9  # reuse
        buf83 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_14.run(buf4, buf7, buf68, buf73, buf79, buf83, 1048576, 49, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf75, (512, 2048, 49), (100352, 49, 1), 0), buf79, buf77, buf78, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf82 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf75, (512, 2048, 49), (100352, 49, 1), 0), buf83, rsqrt_49, primals_298, buf77, buf78, buf82, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del buf75
        del primals_298
        del rsqrt_49
        buf85 = reinterpret_tensor(buf61, (25088, 512), (512, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_627, getitem_628, getitem_629, buf85, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_627
        del getitem_628
        del getitem_629
        buf87 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf88 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf82, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf87, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_50, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_50
        buf89 = buf88[0]
        assert_size_stride(buf89, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf89, 16, 'torch.ops.aten.convolution_backward.default')
        del buf88
        buf90 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf91 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf82, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf85, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf90, (2048, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf92 = buf91[1]
        assert_size_stride(buf92, (2048, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf92, 16, 'torch.ops.aten.convolution_backward.default')
        del buf91
        buf93 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf92, buf93, 1048576, stream=stream0)
        del buf92
        buf94 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        buf115 = empty_strided_cuda((12845056, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(buf29, buf94, buf115, 12845056, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_626, reinterpret_tensor(buf94, (25088, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        del getitem_626
        buf96 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_621, getitem_622, getitem_623, buf96, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_621
        del getitem_622
        del getitem_623
        buf98 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf99 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf119 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf120 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf34, buf98, buf99, buf119, buf120, 512, stream=stream0)
        buf100 = buf62; del buf62  # reuse
        buf104 = reinterpret_tensor(buf54, (512, 512, 49), (25088, 49, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf89, buf94, buf100, buf104, 262144, 49, stream=stream0)
        del buf89
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf96, (512, 512, 49), (25088, 49, 1), 0), buf100, buf98, buf99, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf103 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf96, (512, 512, 49), (25088, 49, 1), 0), buf104, rsqrt_48, primals_292, buf98, buf99, buf103, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del primals_292
        del rsqrt_48
        buf106 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_614, getitem_615, getitem_616, buf106, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_614
        del getitem_615
        del getitem_616
        buf108 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf109 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf103, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf108, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_49
        buf110 = buf109[0]
        assert_size_stride(buf110, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf110, 16, 'torch.ops.aten.convolution_backward.default')
        del buf109
        buf111 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf112 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf103, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf106, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf111, (512, 512, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf113 = buf112[1]
        assert_size_stride(buf113, (512, 512, 3, 3), (4608, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf113, 16, 'torch.ops.aten.convolution_backward.default')
        del buf112
        buf114 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf113, buf114, 2359296, stream=stream0)
        del buf113
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_613, reinterpret_tensor(buf115, (25088, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        del getitem_613
        buf117 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_608, getitem_609, getitem_610, buf117, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_608
        del getitem_609
        del getitem_610
        buf121 = buf103; del buf103  # reuse
        buf125 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf110, buf115, buf121, buf125, 262144, 49, stream=stream0)
        del buf110
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf117, (512, 512, 49), (25088, 49, 1), 0), buf121, buf119, buf120, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf124 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf117, (512, 512, 49), (25088, 49, 1), 0), buf125, rsqrt_47, primals_286, buf119, buf120, buf124, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del primals_286
        del rsqrt_47
        buf127 = reinterpret_tensor(buf82, (100352, 512), (512, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_601, getitem_602, getitem_603, buf127, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_601
        del getitem_602
        del getitem_603
        buf129 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf130 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf124, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf129, (512, 2048, 7, 7), (0, 0, 0, 0), 0), convert_element_type_48, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_48
        buf131 = buf130[0]
        assert_size_stride(buf131, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf131, 16, 'torch.ops.aten.convolution_backward.default')
        del buf130
        buf132 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf133 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf124, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf127, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf132, (512, 2048, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf134 = buf133[1]
        assert_size_stride(buf134, (512, 2048, 1, 1), (2048, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf134, 16, 'torch.ops.aten.convolution_backward.default')
        del buf133
        buf135 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf134, buf135, 1048576, stream=stream0)
        del buf134
        buf136 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf192 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf6, buf136, buf192, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_600, reinterpret_tensor(buf136, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_600
        buf138 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_595, getitem_596, getitem_597, buf138, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_595
        del getitem_596
        del getitem_597
        buf140 = reinterpret_tensor(buf83, (512, 2048, 7, 7), (100352, 49, 7, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_15.run(buf4, buf7, buf68, buf73, buf131, buf136, buf140, 1048576, 49, stream=stream0)
        del buf131
        del buf4
        buf141 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf142 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf158 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(buf11, buf141, buf142, buf158, 2048, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf138, (512, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf140, (512, 2048, 49), (100352, 49, 1), 0), buf141, buf142, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf145 = reinterpret_tensor(buf68, (512, 2048, 49), (100352, 49, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf138, (512, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf140, (512, 2048, 49), (100352, 49, 1), 0), rsqrt_46, primals_280, buf141, buf142, buf145, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del primals_280
        del rsqrt_46
        buf147 = empty_strided_cuda((200704, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_588, getitem_589, getitem_590, buf147, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_588
        del getitem_589
        del getitem_590
        buf149 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf150 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf145, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf149, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_47, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_47
        buf151 = buf150[0]
        assert_size_stride(buf151, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf151, 16, 'torch.ops.aten.convolution_backward.default')
        del buf150
        buf152 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf153 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf145, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf147, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf152, (2048, 1024, 1, 1), (0, 0, 0, 0), 0), None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf154 = buf153[1]
        assert_size_stride(buf154, (2048, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf154, 16, 'torch.ops.aten.convolution_backward.default')
        del buf153
        buf155 = empty_strided_cuda((2048, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_17.run(buf154, buf155, 2097152, stream=stream0)
        del buf154
        buf156 = reinterpret_tensor(buf145, (100352, 512), (512, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_585, getitem_586, getitem_587, buf156, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_585
        del getitem_586
        del getitem_587
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_2.run(reinterpret_tensor(buf156, (512, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf140, (512, 2048, 49), (100352, 49, 1), 0), buf11, buf158, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        buf161 = reinterpret_tensor(buf138, (512, 2048, 49), (100352, 49, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_3.run(reinterpret_tensor(buf156, (512, 2048, 49), (100352, 49, 1), 0), reinterpret_tensor(buf140, (512, 2048, 49), (100352, 49, 1), 0), rsqrt_45, primals_274, buf11, buf158, buf161, 25088, 49, 100352, 49, 1024, 2048, 25, 1, stream=stream0)
        del primals_274
        del rsqrt_45
        buf163 = reinterpret_tensor(buf124, (25088, 512), (512, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_578, getitem_579, getitem_580, buf163, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_578
        del getitem_579
        del getitem_580
        buf165 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf166 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf161, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf165, (512, 512, 7, 7), (0, 0, 0, 0), 0), convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_46
        buf167 = buf166[0]
        assert_size_stride(buf167, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf167, 16, 'torch.ops.aten.convolution_backward.default')
        del buf166
        buf168 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf169 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf161, (512, 2048, 7, 7), (100352, 49, 7, 1), 0), reinterpret_tensor(buf163, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf168, (2048, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf170 = buf169[1]
        assert_size_stride(buf170, (2048, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf170, 16, 'torch.ops.aten.convolution_backward.default')
        del buf169
        buf171 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf170, buf171, 1048576, stream=stream0)
        del buf170
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_577, buf29, 16, 1, 512, 1, 1, 32, 16, 25088, 1, 1, stream=stream0)
        del buf115
        del buf30
        del buf52
        del buf94
        del getitem_577
        buf173 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_572, getitem_573, getitem_574, buf173, 32, 1, 512, 1, 2, 16, 32, 25088, 1, 1, stream=stream0)
        del getitem_572
        del getitem_573
        del getitem_574
        buf175 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf176 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf196 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf197 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf34, buf175, buf176, buf196, buf197, 512, stream=stream0)
        buf177 = buf125; del buf125  # reuse
        buf181 = reinterpret_tensor(buf117, (512, 512, 49), (25088, 49, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(buf167, buf29, buf177, buf181, 262144, 49, stream=stream0)
        del buf167
        del buf29
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_4.run(reinterpret_tensor(buf173, (512, 512, 49), (25088, 49, 1), 0), buf177, buf175, buf176, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        buf180 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_5.run(reinterpret_tensor(buf173, (512, 512, 49), (25088, 49, 1), 0), buf181, rsqrt_44, primals_268, buf175, buf176, buf180, 25088, 49, 25088, 49, 1024, 512, 25, 1, stream=stream0)
        del buf173
        del buf181
        del primals_268
        del rsqrt_44
        buf183 = reinterpret_tensor(buf161, (100352, 512), (512, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_565, getitem_566, getitem_567, buf183, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_565
        del getitem_566
        del getitem_567
        buf185 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf186 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf180, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf185, (512, 512, 14, 14), (0, 0, 0, 0), 0), convert_element_type_45, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_45
        buf187 = buf186[0]
        assert_size_stride(buf187, (512, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf187, 16, 'torch.ops.aten.convolution_backward.default')
        del buf186
        buf188 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf189 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf180, (512, 512, 7, 7), (25088, 49, 7, 1), 0), reinterpret_tensor(buf183, (512, 512, 14, 14), (100352, 196, 14, 1), 0), reinterpret_tensor(buf188, (512, 512, 3, 3), (0, 0, 0, 0), 0), None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        del buf180
        buf190 = buf189[1]
        assert_size_stride(buf190, (512, 512, 3, 3), (4608, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf190, 16, 'torch.ops.aten.convolution_backward.default')
        del buf189
        buf191 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(buf190, buf191, 2359296, stream=stream0)
        del buf190
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_564, reinterpret_tensor(buf192, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_564
        buf194 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_559, getitem_560, getitem_561, buf194, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_559
        del getitem_560
        del getitem_561
        buf198 = reinterpret_tensor(buf156, (512, 512, 196), (100352, 196, 1), 0); del buf156  # reuse
        buf202 = reinterpret_tensor(buf140, (512, 512, 196), (100352, 196, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_18.run(buf187, buf192, buf198, buf202, 262144, 196, stream=stream0)
        del buf187
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_6.run(reinterpret_tensor(buf194, (512, 512, 196), (100352, 196, 1), 0), buf198, buf196, buf197, 100352, 196, 100352, 196, 1024, 512, 98, 1, stream=stream0)
        buf201 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_7.run(reinterpret_tensor(buf194, (512, 512, 196), (100352, 196, 1), 0), buf202, rsqrt_43, primals_262, buf196, buf197, buf201, 100352, 196, 100352, 196, 1024, 512, 98, 1, stream=stream0)
        del primals_262
        del rsqrt_43
        buf204 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_552, getitem_553, getitem_554, buf204, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_552
        del getitem_553
        del getitem_554
        buf206 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf207 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf201, (512, 512, 14, 14), (100352, 196, 14, 1), 0), reinterpret_tensor(buf206, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_44, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_44
        buf208 = buf207[0]
        assert_size_stride(buf208, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf208, 16, 'torch.ops.aten.convolution_backward.default')
        del buf207
        buf209 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf210 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf201, (512, 512, 14, 14), (100352, 196, 14, 1), 0), reinterpret_tensor(buf204, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf209, (512, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf211 = buf210[1]
        assert_size_stride(buf211, (512, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf211, 16, 'torch.ops.aten.convolution_backward.default')
        del buf210
        buf212 = empty_strided_cuda((512, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_19.run(buf211, buf212, 524288, stream=stream0)
        del buf211
        buf213 = empty_strided_cuda((200704, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_286], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_20.run(buf213, 102760448, stream=stream0)
        buf214 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_21.run(buf213, buf214, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_551, reinterpret_tensor(buf214, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_551
        buf216 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_546, getitem_547, getitem_548, buf216, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_546
        del getitem_547
        del getitem_548
        buf218 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer3_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_22.run(buf218, 1024, stream=stream0)
        buf219 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf220 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_23.run(buf218, buf219, buf220, 1024, stream=stream0)
        buf221 = empty_strided_cuda((512, 1024, 196), (200704, 196, 1), torch.bfloat16)
        buf225 = empty_strided_cuda((512, 1024, 196), (200704, 196, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_24.run(buf151, buf208, buf214, buf221, buf225, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf216, (512, 1024, 196), (200704, 196, 1), 0), buf221, buf219, buf220, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf224 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf216, (512, 1024, 196), (200704, 196, 1), 0), buf225, rsqrt_42, primals_256, buf219, buf220, buf224, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del buf216
        del buf225
        del primals_256
        del rsqrt_42
        buf227 = empty_strided_cuda((50176, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_539, getitem_540, getitem_541, buf227, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_539
        del getitem_540
        del getitem_541
        buf229 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf230 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf224, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf229, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_43, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_43
        buf231 = buf230[0]
        assert_size_stride(buf231, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf231, 16, 'torch.ops.aten.convolution_backward.default')
        del buf230
        buf232 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf233 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf224, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf227, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf232, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf234 = buf233[1]
        assert_size_stride(buf234, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf234, 16, 'torch.ops.aten.convolution_backward.default')
        del buf233
        buf235 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf234, buf235, 262144, stream=stream0)
        del buf234
        buf236 = empty_strided_cuda((50176, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_289], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_26.run(buf236, 25690112, stream=stream0)
        buf237 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf259 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_27.run(buf236, buf237, buf259, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_538, reinterpret_tensor(buf237, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_538
        buf239 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_533, getitem_534, getitem_535, buf239, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_533
        del getitem_534
        del getitem_535
        buf241 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer1_0_bn3], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_28.run(buf241, 256, stream=stream0)
        buf242 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf243 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf263 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf264 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf241, buf242, buf243, buf263, buf264, 256, stream=stream0)
        buf244 = empty_strided_cuda((512, 256, 196), (50176, 196, 1), torch.bfloat16)
        buf248 = empty_strided_cuda((512, 256, 196), (50176, 196, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf231, buf237, buf244, buf248, 131072, 196, stream=stream0)
        del buf231
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf239, (512, 256, 196), (50176, 196, 1), 0), buf244, buf242, buf243, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf247 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf239, (512, 256, 196), (50176, 196, 1), 0), buf248, rsqrt_41, primals_250, buf242, buf243, buf247, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_250
        del rsqrt_41
        buf250 = reinterpret_tensor(buf248, (50176, 512), (512, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_526, getitem_527, getitem_528, buf250, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_526
        del getitem_527
        del getitem_528
        buf252 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf253 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf247, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf252, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_42, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_42
        buf254 = buf253[0]
        assert_size_stride(buf254, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf254, 16, 'torch.ops.aten.convolution_backward.default')
        del buf253
        buf255 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf256 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf247, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf250, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf255, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf257 = buf256[1]
        assert_size_stride(buf257, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf257, 16, 'torch.ops.aten.convolution_backward.default')
        del buf256
        buf258 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_31.run(buf257, buf258, 589824, stream=stream0)
        del buf257
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_525, reinterpret_tensor(buf259, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_525
        buf261 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_520, getitem_521, getitem_522, buf261, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_520
        del getitem_521
        del getitem_522
        buf265 = buf247; del buf247  # reuse
        buf269 = reinterpret_tensor(buf239, (512, 256, 196), (50176, 196, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf254, buf259, buf265, buf269, 131072, 196, stream=stream0)
        del buf254
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf261, (512, 256, 196), (50176, 196, 1), 0), buf265, buf263, buf264, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf268 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf261, (512, 256, 196), (50176, 196, 1), 0), buf269, rsqrt_40, primals_244, buf263, buf264, buf268, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_244
        del rsqrt_40
        buf271 = reinterpret_tensor(buf224, (200704, 512), (512, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_513, getitem_514, getitem_515, buf271, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_513
        del getitem_514
        del getitem_515
        buf273 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf274 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf268, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf273, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_41
        buf275 = buf274[0]
        assert_size_stride(buf275, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf275, 16, 'torch.ops.aten.convolution_backward.default')
        del buf274
        buf276 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf277 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf268, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf271, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf276, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf278 = buf277[1]
        assert_size_stride(buf278, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf278, 16, 'torch.ops.aten.convolution_backward.default')
        del buf277
        buf279 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf278, buf279, 262144, stream=stream0)
        del buf278
        buf280 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf342 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf213, buf280, buf342, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_512, reinterpret_tensor(buf280, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_512
        buf282 = reinterpret_tensor(buf271, (512, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_33.run(buf151, buf208, buf214, buf275, buf280, buf282, 524288, 196, stream=stream0)
        buf283 = reinterpret_tensor(buf275, (200704, 512), (512, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_507, getitem_508, getitem_509, buf283, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_507
        del getitem_508
        del getitem_509
        buf285 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf286 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf346 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf347 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf218, buf285, buf286, buf346, buf347, 1024, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf283, (512, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf282, (512, 1024, 196), (200704, 196, 1), 0), buf285, buf286, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf289 = reinterpret_tensor(buf208, (512, 1024, 196), (200704, 196, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf283, (512, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf282, (512, 1024, 196), (200704, 196, 1), 0), rsqrt_39, primals_238, buf285, buf286, buf289, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_238
        del rsqrt_39
        buf291 = reinterpret_tensor(buf268, (50176, 512), (512, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_500, getitem_501, getitem_502, buf291, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_500
        del getitem_501
        del getitem_502
        buf293 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf294 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf289, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf293, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_40
        buf295 = buf294[0]
        assert_size_stride(buf295, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf295, 16, 'torch.ops.aten.convolution_backward.default')
        del buf294
        buf296 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf297 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf289, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf291, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf296, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf298 = buf297[1]
        assert_size_stride(buf298, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf298, 16, 'torch.ops.aten.convolution_backward.default')
        del buf297
        buf299 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf298, buf299, 262144, stream=stream0)
        del buf298
        buf300 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf321 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_27.run(buf236, buf300, buf321, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_499, reinterpret_tensor(buf300, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_499
        buf302 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_494, getitem_495, getitem_496, buf302, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_494
        del getitem_495
        del getitem_496
        buf304 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf305 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf325 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf326 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf241, buf304, buf305, buf325, buf326, 256, stream=stream0)
        buf306 = buf269; del buf269  # reuse
        buf310 = reinterpret_tensor(buf261, (512, 256, 196), (50176, 196, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf295, buf300, buf306, buf310, 131072, 196, stream=stream0)
        del buf295
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf302, (512, 256, 196), (50176, 196, 1), 0), buf306, buf304, buf305, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf309 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf302, (512, 256, 196), (50176, 196, 1), 0), buf310, rsqrt_38, primals_232, buf304, buf305, buf309, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_232
        del rsqrt_38
        buf312 = reinterpret_tensor(buf310, (50176, 512), (512, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_487, getitem_488, getitem_489, buf312, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_487
        del getitem_488
        del getitem_489
        buf314 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf315 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf309, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf314, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_39
        buf316 = buf315[0]
        assert_size_stride(buf316, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf316, 16, 'torch.ops.aten.convolution_backward.default')
        del buf315
        buf317 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf318 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf309, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf312, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf317, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf319 = buf318[1]
        assert_size_stride(buf319, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf319, 16, 'torch.ops.aten.convolution_backward.default')
        del buf318
        buf320 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_31.run(buf319, buf320, 589824, stream=stream0)
        del buf319
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_486, reinterpret_tensor(buf321, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_486
        buf323 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_481, getitem_482, getitem_483, buf323, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_481
        del getitem_482
        del getitem_483
        buf327 = buf309; del buf309  # reuse
        buf331 = reinterpret_tensor(buf302, (512, 256, 196), (50176, 196, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf316, buf321, buf327, buf331, 131072, 196, stream=stream0)
        del buf316
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf323, (512, 256, 196), (50176, 196, 1), 0), buf327, buf325, buf326, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf330 = buf327; del buf327  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf323, (512, 256, 196), (50176, 196, 1), 0), buf331, rsqrt_37, primals_226, buf325, buf326, buf330, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_226
        del rsqrt_37
        buf333 = reinterpret_tensor(buf289, (200704, 512), (512, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_474, getitem_475, getitem_476, buf333, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_474
        del getitem_475
        del getitem_476
        buf335 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf336 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf330, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf335, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_38, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_38
        buf337 = buf336[0]
        assert_size_stride(buf337, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf337, 16, 'torch.ops.aten.convolution_backward.default')
        del buf336
        buf338 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf339 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf330, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf333, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf338, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf340 = buf339[1]
        assert_size_stride(buf340, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf340, 16, 'torch.ops.aten.convolution_backward.default')
        del buf339
        buf341 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf340, buf341, 262144, stream=stream0)
        del buf340
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_473, reinterpret_tensor(buf342, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_473
        buf344 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_468, getitem_469, getitem_470, buf344, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_468
        del getitem_469
        del getitem_470
        buf348 = reinterpret_tensor(buf283, (512, 1024, 196), (200704, 196, 1), 0); del buf283  # reuse
        buf352 = reinterpret_tensor(buf151, (512, 1024, 196), (200704, 196, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf282, buf337, buf342, buf348, buf352, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf344, (512, 1024, 196), (200704, 196, 1), 0), buf348, buf346, buf347, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf351 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf344, (512, 1024, 196), (200704, 196, 1), 0), buf352, rsqrt_36, primals_220, buf346, buf347, buf351, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_220
        del rsqrt_36
        buf354 = reinterpret_tensor(buf330, (50176, 512), (512, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_461, getitem_462, getitem_463, buf354, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_461
        del getitem_462
        del getitem_463
        buf356 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf357 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf351, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf356, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_37
        buf358 = buf357[0]
        assert_size_stride(buf358, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf358, 16, 'torch.ops.aten.convolution_backward.default')
        del buf357
        buf359 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf360 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf351, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf354, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf359, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf361 = buf360[1]
        assert_size_stride(buf361, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf361, 16, 'torch.ops.aten.convolution_backward.default')
        del buf360
        buf362 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf361, buf362, 262144, stream=stream0)
        del buf361
        buf363 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf384 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_27.run(buf236, buf363, buf384, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_460, reinterpret_tensor(buf363, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_460
        buf365 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_455, getitem_456, getitem_457, buf365, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_455
        del getitem_456
        del getitem_457
        buf367 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf368 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf388 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf389 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf241, buf367, buf368, buf388, buf389, 256, stream=stream0)
        buf369 = buf331; del buf331  # reuse
        buf373 = reinterpret_tensor(buf323, (512, 256, 196), (50176, 196, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf358, buf363, buf369, buf373, 131072, 196, stream=stream0)
        del buf358
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf365, (512, 256, 196), (50176, 196, 1), 0), buf369, buf367, buf368, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf372 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf365, (512, 256, 196), (50176, 196, 1), 0), buf373, rsqrt_35, primals_214, buf367, buf368, buf372, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_214
        del rsqrt_35
        buf375 = reinterpret_tensor(buf373, (50176, 512), (512, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_448, getitem_449, getitem_450, buf375, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_448
        del getitem_449
        del getitem_450
        buf377 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf378 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf372, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf377, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_36, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_36
        buf379 = buf378[0]
        assert_size_stride(buf379, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf379, 16, 'torch.ops.aten.convolution_backward.default')
        del buf378
        buf380 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf381 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf372, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf375, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf380, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf382 = buf381[1]
        assert_size_stride(buf382, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf382, 16, 'torch.ops.aten.convolution_backward.default')
        del buf381
        buf383 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_31.run(buf382, buf383, 589824, stream=stream0)
        del buf382
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_447, reinterpret_tensor(buf384, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_447
        buf386 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_442, getitem_443, getitem_444, buf386, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_442
        del getitem_443
        del getitem_444
        buf390 = buf372; del buf372  # reuse
        buf394 = reinterpret_tensor(buf365, (512, 256, 196), (50176, 196, 1), 0); del buf365  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf379, buf384, buf390, buf394, 131072, 196, stream=stream0)
        del buf379
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf386, (512, 256, 196), (50176, 196, 1), 0), buf390, buf388, buf389, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf393 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf386, (512, 256, 196), (50176, 196, 1), 0), buf394, rsqrt_34, primals_208, buf388, buf389, buf393, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_208
        del rsqrt_34
        buf396 = reinterpret_tensor(buf351, (200704, 512), (512, 1), 0); del buf351  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_435, getitem_436, getitem_437, buf396, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_435
        del getitem_436
        del getitem_437
        buf398 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf399 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf393, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf398, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_35, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_35
        buf400 = buf399[0]
        assert_size_stride(buf400, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf400, 16, 'torch.ops.aten.convolution_backward.default')
        del buf399
        buf401 = buf398; del buf398  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf402 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf393, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf396, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf401, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf403 = buf402[1]
        assert_size_stride(buf403, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf403, 16, 'torch.ops.aten.convolution_backward.default')
        del buf402
        buf404 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf403, buf404, 262144, stream=stream0)
        del buf403
        buf405 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf467 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf213, buf405, buf467, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_434, reinterpret_tensor(buf405, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_434
        buf407 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_36.run(buf407, buf337, buf342, buf400, buf405, 524288, 196, stream=stream0)
        buf408 = reinterpret_tensor(buf400, (200704, 512), (512, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_429, getitem_430, getitem_431, buf408, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_429
        del getitem_430
        del getitem_431
        buf410 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf411 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf471 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf472 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_34.run(buf218, buf410, buf411, buf471, buf472, 1024, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf408, (512, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf407, (512, 1024, 196), (200704, 196, 1), 0), buf410, buf411, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf414 = reinterpret_tensor(buf337, (512, 1024, 196), (200704, 196, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf408, (512, 1024, 196), (200704, 196, 1), 0), reinterpret_tensor(buf407, (512, 1024, 196), (200704, 196, 1), 0), rsqrt_33, primals_202, buf410, buf411, buf414, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_202
        del rsqrt_33
        buf416 = reinterpret_tensor(buf393, (50176, 512), (512, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_422, getitem_423, getitem_424, buf416, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_422
        del getitem_423
        del getitem_424
        buf418 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf419 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf414, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf418, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_34
        buf420 = buf419[0]
        assert_size_stride(buf420, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf420, 16, 'torch.ops.aten.convolution_backward.default')
        del buf419
        buf421 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf422 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf414, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf416, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf421, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf423 = buf422[1]
        assert_size_stride(buf423, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf423, 16, 'torch.ops.aten.convolution_backward.default')
        del buf422
        buf424 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf423, buf424, 262144, stream=stream0)
        del buf423
        buf425 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf446 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_27.run(buf236, buf425, buf446, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_421, reinterpret_tensor(buf425, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_421
        buf427 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_416, getitem_417, getitem_418, buf427, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_416
        del getitem_417
        del getitem_418
        buf429 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf430 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf450 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf451 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf241, buf429, buf430, buf450, buf451, 256, stream=stream0)
        buf431 = buf394; del buf394  # reuse
        buf435 = reinterpret_tensor(buf386, (512, 256, 196), (50176, 196, 1), 0); del buf386  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf420, buf425, buf431, buf435, 131072, 196, stream=stream0)
        del buf420
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf427, (512, 256, 196), (50176, 196, 1), 0), buf431, buf429, buf430, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf434 = buf431; del buf431  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf427, (512, 256, 196), (50176, 196, 1), 0), buf435, rsqrt_32, primals_196, buf429, buf430, buf434, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_196
        del rsqrt_32
        buf437 = reinterpret_tensor(buf435, (50176, 512), (512, 1), 0); del buf435  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_409, getitem_410, getitem_411, buf437, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_409
        del getitem_410
        del getitem_411
        buf439 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf440 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf434, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf439, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_33
        buf441 = buf440[0]
        assert_size_stride(buf441, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf441, 16, 'torch.ops.aten.convolution_backward.default')
        del buf440
        buf442 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf443 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf434, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf437, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf442, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf444 = buf443[1]
        assert_size_stride(buf444, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf444, 16, 'torch.ops.aten.convolution_backward.default')
        del buf443
        buf445 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_31.run(buf444, buf445, 589824, stream=stream0)
        del buf444
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_408, reinterpret_tensor(buf446, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_408
        buf448 = buf437; del buf437  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_403, getitem_404, getitem_405, buf448, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_403
        del getitem_404
        del getitem_405
        buf452 = buf434; del buf434  # reuse
        buf456 = reinterpret_tensor(buf427, (512, 256, 196), (50176, 196, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf441, buf446, buf452, buf456, 131072, 196, stream=stream0)
        del buf441
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf448, (512, 256, 196), (50176, 196, 1), 0), buf452, buf450, buf451, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf455 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf448, (512, 256, 196), (50176, 196, 1), 0), buf456, rsqrt_31, primals_190, buf450, buf451, buf455, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_190
        del rsqrt_31
        buf458 = reinterpret_tensor(buf414, (200704, 512), (512, 1), 0); del buf414  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_396, getitem_397, getitem_398, buf458, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_396
        del getitem_397
        del getitem_398
        buf460 = buf442; del buf442  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf461 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf455, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf460, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_32, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_32
        buf462 = buf461[0]
        assert_size_stride(buf462, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf462, 16, 'torch.ops.aten.convolution_backward.default')
        del buf461
        buf463 = buf460; del buf460  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf464 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf455, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf458, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf463, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf465 = buf464[1]
        assert_size_stride(buf465, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf465, 16, 'torch.ops.aten.convolution_backward.default')
        del buf464
        buf466 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf465, buf466, 262144, stream=stream0)
        del buf465
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_395, reinterpret_tensor(buf467, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_395
        buf469 = buf458; del buf458  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_390, getitem_391, getitem_392, buf469, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_390
        del getitem_391
        del getitem_392
        buf473 = reinterpret_tensor(buf408, (512, 1024, 196), (200704, 196, 1), 0); del buf408  # reuse
        buf477 = reinterpret_tensor(buf396, (512, 1024, 196), (200704, 196, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_35.run(buf407, buf462, buf467, buf473, buf477, 524288, 196, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf469, (512, 1024, 196), (200704, 196, 1), 0), buf473, buf471, buf472, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf476 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf469, (512, 1024, 196), (200704, 196, 1), 0), buf477, rsqrt_30, primals_184, buf471, buf472, buf476, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_184
        del rsqrt_30
        buf479 = reinterpret_tensor(buf455, (50176, 512), (512, 1), 0); del buf455  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_383, getitem_384, getitem_385, buf479, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_383
        del getitem_384
        del getitem_385
        buf481 = buf463; del buf463  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf482 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf476, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf481, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_31
        buf483 = buf482[0]
        assert_size_stride(buf483, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf483, 16, 'torch.ops.aten.convolution_backward.default')
        del buf482
        buf484 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf485 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf476, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf479, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf484, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf486 = buf485[1]
        assert_size_stride(buf486, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf486, 16, 'torch.ops.aten.convolution_backward.default')
        del buf485
        buf487 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf486, buf487, 262144, stream=stream0)
        del buf486
        buf488 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        buf509 = empty_strided_cuda((25690112, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_27.run(buf236, buf488, buf509, 25690112, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_382, reinterpret_tensor(buf488, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_382
        buf490 = buf479; del buf479  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_377, getitem_378, getitem_379, buf490, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_377
        del getitem_378
        del getitem_379
        buf492 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf493 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf513 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf514 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf241, buf492, buf493, buf513, buf514, 256, stream=stream0)
        buf494 = buf456; del buf456  # reuse
        buf498 = reinterpret_tensor(buf448, (512, 256, 196), (50176, 196, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf483, buf488, buf494, buf498, 131072, 196, stream=stream0)
        del buf483
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf490, (512, 256, 196), (50176, 196, 1), 0), buf494, buf492, buf493, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf497 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf490, (512, 256, 196), (50176, 196, 1), 0), buf498, rsqrt_29, primals_178, buf492, buf493, buf497, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_178
        del rsqrt_29
        buf500 = reinterpret_tensor(buf498, (50176, 512), (512, 1), 0); del buf498  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_370, getitem_371, getitem_372, buf500, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_370
        del getitem_371
        del getitem_372
        buf502 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf503 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf497, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf502, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_30, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_30
        buf504 = buf503[0]
        assert_size_stride(buf504, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf504, 16, 'torch.ops.aten.convolution_backward.default')
        del buf503
        buf505 = buf502; del buf502  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf506 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf497, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf500, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf505, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf507 = buf506[1]
        assert_size_stride(buf507, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf507, 16, 'torch.ops.aten.convolution_backward.default')
        del buf506
        buf508 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_31.run(buf507, buf508, 589824, stream=stream0)
        del buf507
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_369, reinterpret_tensor(buf509, (50176, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del getitem_369
        buf511 = buf500; del buf500  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_364, getitem_365, getitem_366, buf511, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_364
        del getitem_365
        del getitem_366
        buf515 = buf497; del buf497  # reuse
        buf519 = reinterpret_tensor(buf490, (512, 256, 196), (50176, 196, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf504, buf509, buf515, buf519, 131072, 196, stream=stream0)
        del buf504
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf511, (512, 256, 196), (50176, 196, 1), 0), buf515, buf513, buf514, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf518 = buf515; del buf515  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf511, (512, 256, 196), (50176, 196, 1), 0), buf519, rsqrt_28, primals_172, buf513, buf514, buf518, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del primals_172
        del rsqrt_28
        buf521 = reinterpret_tensor(buf476, (200704, 512), (512, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_357, getitem_358, getitem_359, buf521, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_357
        del getitem_358
        del getitem_359
        buf523 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf524 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf518, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf523, (512, 1024, 14, 14), (0, 0, 0, 0), 0), convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_29
        buf525 = buf524[0]
        assert_size_stride(buf525, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf525, 16, 'torch.ops.aten.convolution_backward.default')
        del buf524
        buf526 = buf523; del buf523  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf527 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf518, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf521, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf526, (256, 1024, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf528 = buf527[1]
        assert_size_stride(buf528, (256, 1024, 1, 1), (1024, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf528, 16, 'torch.ops.aten.convolution_backward.default')
        del buf527
        buf529 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf528, buf529, 262144, stream=stream0)
        del buf528
        buf530 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf590 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf213, buf530, buf590, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_356, reinterpret_tensor(buf530, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_356
        buf532 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_351, getitem_352, getitem_353, buf532, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_351
        del getitem_352
        del getitem_353
        buf534 = buf462; del buf462  # reuse
        buf537 = buf477; del buf477  # reuse
        buf541 = reinterpret_tensor(buf469, (512, 1024, 196), (200704, 196, 1), 0); del buf469  # reuse
        buf555 = buf352; del buf352  # reuse
        buf559 = reinterpret_tensor(buf344, (512, 1024, 196), (200704, 196, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_37.run(buf534, buf407, buf467, buf525, buf530, buf537, buf541, buf555, buf559, 524288, 196, stream=stream0)
        del buf407
        del buf525
        del buf534
        buf535 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf536 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf554 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_38.run(buf218, buf535, buf536, buf554, 1024, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf532, (512, 1024, 196), (200704, 196, 1), 0), buf537, buf535, buf536, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf540 = buf537; del buf537  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf532, (512, 1024, 196), (200704, 196, 1), 0), buf541, rsqrt_27, primals_166, buf535, buf536, buf540, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del buf532
        del buf541
        del primals_166
        del rsqrt_27
        buf543 = empty_strided_cuda((401408, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_344, getitem_345, getitem_346, buf543, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_344
        del getitem_345
        del getitem_346
        buf545 = buf526; del buf526  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf546 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf540, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf545, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_28, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_28
        buf547 = buf546[0]
        assert_size_stride(buf547, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf547, 16, 'torch.ops.aten.convolution_backward.default')
        del buf546
        buf548 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf549 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf540, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf543, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf548, (1024, 512, 1, 1), (0, 0, 0, 0), 0), None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf550 = buf549[1]
        assert_size_stride(buf550, (1024, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf550, 16, 'torch.ops.aten.convolution_backward.default')
        del buf549
        buf551 = empty_strided_cuda((1024, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_19.run(buf550, buf551, 524288, stream=stream0)
        del buf550
        buf552 = reinterpret_tensor(buf540, (200704, 512), (512, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_341, getitem_342, getitem_343, buf552, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_341
        del getitem_342
        del getitem_343
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_8.run(reinterpret_tensor(buf552, (512, 1024, 196), (200704, 196, 1), 0), buf555, buf218, buf554, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        buf558 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_9.run(reinterpret_tensor(buf552, (512, 1024, 196), (200704, 196, 1), 0), buf559, rsqrt_26, primals_160, buf218, buf554, buf558, 100352, 196, 200704, 196, 1024, 1024, 98, 1, stream=stream0)
        del primals_160
        del rsqrt_26
        buf561 = reinterpret_tensor(buf518, (50176, 512), (512, 1), 0); del buf518  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_334, getitem_335, getitem_336, buf561, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_334
        del getitem_335
        del getitem_336
        buf563 = buf548; del buf548  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf564 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf558, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf563, (512, 256, 14, 14), (0, 0, 0, 0), 0), convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_27
        buf565 = buf564[0]
        assert_size_stride(buf565, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf565, 16, 'torch.ops.aten.convolution_backward.default')
        del buf564
        buf566 = buf563; del buf563  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf567 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf558, (512, 1024, 14, 14), (200704, 196, 14, 1), 0), reinterpret_tensor(buf561, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf566, (1024, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf568 = buf567[1]
        assert_size_stride(buf568, (1024, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf568, 16, 'torch.ops.aten.convolution_backward.default')
        del buf567
        buf569 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_25.run(buf568, buf569, 262144, stream=stream0)
        del buf568
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_333, buf236, 16, 1, 512, 1, 1, 32, 16, 50176, 1, 1, stream=stream0)
        del buf237
        del buf259
        del buf300
        del buf321
        del buf363
        del buf384
        del buf425
        del buf446
        del buf488
        del buf509
        del getitem_333
        buf571 = buf561; del buf561  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_328, getitem_329, getitem_330, buf571, 32, 1, 512, 1, 2, 16, 32, 50176, 1, 1, stream=stream0)
        del getitem_328
        del getitem_329
        del getitem_330
        buf573 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf574 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf594 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf595 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf241, buf573, buf574, buf594, buf595, 256, stream=stream0)
        buf575 = buf519; del buf519  # reuse
        buf579 = reinterpret_tensor(buf511, (512, 256, 196), (50176, 196, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_30.run(buf565, buf236, buf575, buf579, 131072, 196, stream=stream0)
        del buf236
        del buf565
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_10.run(reinterpret_tensor(buf571, (512, 256, 196), (50176, 196, 1), 0), buf575, buf573, buf574, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        buf578 = buf575; del buf575  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_11.run(reinterpret_tensor(buf571, (512, 256, 196), (50176, 196, 1), 0), buf579, rsqrt_25, primals_154, buf573, buf574, buf578, 100352, 196, 50176, 196, 1024, 256, 98, 1, stream=stream0)
        del buf571
        del buf579
        del primals_154
        del rsqrt_25
        buf581 = reinterpret_tensor(buf558, (200704, 512), (512, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_321, getitem_322, getitem_323, buf581, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_321
        del getitem_322
        del getitem_323
        buf583 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf584 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf578, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf583, (512, 256, 28, 28), (0, 0, 0, 0), 0), convert_element_type_26, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_26
        buf585 = buf584[0]
        assert_size_stride(buf585, (512, 256, 28, 28), (200704, 1, 7168, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf585, 16, 'torch.ops.aten.convolution_backward.default')
        del buf584
        buf586 = buf583; del buf583  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf587 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf578, (512, 256, 14, 14), (50176, 196, 14, 1), 0), reinterpret_tensor(buf581, (512, 256, 28, 28), (200704, 784, 28, 1), 0), reinterpret_tensor(buf586, (256, 256, 3, 3), (0, 0, 0, 0), 0), None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        del buf578
        buf588 = buf587[1]
        assert_size_stride(buf588, (256, 256, 3, 3), (2304, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf588, 16, 'torch.ops.aten.convolution_backward.default')
        del buf587
        buf589 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_31.run(buf588, buf589, 589824, stream=stream0)
        del buf588
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_320, reinterpret_tensor(buf590, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_320
        buf592 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_315, getitem_316, getitem_317, buf592, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_315
        del getitem_316
        del getitem_317
        buf596 = reinterpret_tensor(buf559, (512, 256, 784), (200704, 784, 1), 0); del buf559  # reuse
        buf600 = reinterpret_tensor(buf552, (512, 256, 784), (200704, 784, 1), 0); del buf552  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_39.run(buf585, buf590, buf596, buf600, 131072, 784, stream=stream0)
        del buf585
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_12.run(reinterpret_tensor(buf592, (512, 256, 784), (200704, 784, 1), 0), buf596, buf594, buf595, 401408, 784, 200704, 784, 1024, 256, 392, 1, stream=stream0)
        buf599 = buf596; del buf596  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_13.run(reinterpret_tensor(buf592, (512, 256, 784), (200704, 784, 1), 0), buf600, rsqrt_24, primals_148, buf594, buf595, buf599, 401408, 784, 200704, 784, 1024, 256, 392, 1, stream=stream0)
        del primals_148
        del rsqrt_24
        buf602 = buf543; del buf543  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_308, getitem_309, getitem_310, buf602, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_308
        del getitem_309
        del getitem_310
        buf604 = buf586; del buf586  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf605 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf599, (512, 256, 28, 28), (200704, 784, 28, 1), 0), reinterpret_tensor(buf604, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_25
        buf606 = buf605[0]
        assert_size_stride(buf606, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf606, 16, 'torch.ops.aten.convolution_backward.default')
        del buf605
        buf607 = buf604; del buf604  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf608 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf599, (512, 256, 28, 28), (200704, 784, 28, 1), 0), reinterpret_tensor(buf602, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf607, (256, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf609 = buf608[1]
        assert_size_stride(buf609, (256, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf609, 16, 'torch.ops.aten.convolution_backward.default')
        del buf608
        buf610 = empty_strided_cuda((256, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(buf609, buf610, 131072, stream=stream0)
        del buf609
        buf611 = empty_strided_cuda((401408, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_342], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_41.run(buf611, 205520896, stream=stream0)
        buf612 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_42.run(buf611, buf612, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_307, reinterpret_tensor(buf612, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_307
        buf614 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_302, getitem_303, getitem_304, buf614, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_302
        del getitem_303
        del getitem_304
        buf616 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf617 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_43.run(buf34, buf616, buf617, 512, stream=stream0)
        buf618 = empty_strided_cuda((512, 512, 784), (401408, 784, 1), torch.bfloat16)
        buf622 = empty_strided_cuda((512, 512, 784), (401408, 784, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_44.run(buf547, buf606, buf612, buf618, buf622, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf614, (512, 512, 784), (401408, 784, 1), 0), buf618, buf616, buf617, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf621 = buf618; del buf618  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf614, (512, 512, 784), (401408, 784, 1), 0), buf622, rsqrt_23, primals_142, buf616, buf617, buf621, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_142
        del rsqrt_23
        buf624 = reinterpret_tensor(buf201, (100352, 512), (512, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_295, getitem_296, getitem_297, buf624, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_295
        del getitem_296
        del getitem_297
        buf626 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf627 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf621, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf626, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_24
        buf628 = buf627[0]
        assert_size_stride(buf628, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf628, 16, 'torch.ops.aten.convolution_backward.default')
        del buf627
        buf629 = buf626; del buf626  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf630 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf621, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf624, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf629, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf631 = buf630[1]
        assert_size_stride(buf631, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf631, 16, 'torch.ops.aten.convolution_backward.default')
        del buf630
        buf632 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_45.run(buf631, buf632, 65536, stream=stream0)
        del buf631
        buf633 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf655 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf6, buf633, buf655, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_294, reinterpret_tensor(buf633, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_294
        buf635 = buf624; del buf624  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_289, getitem_290, getitem_291, buf635, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_289
        del getitem_290
        del getitem_291
        buf637 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer2_0_bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_46.run(buf637, 128, stream=stream0)
        buf638 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf639 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf659 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf660 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf637, buf638, buf639, buf659, buf660, 128, stream=stream0)
        buf640 = reinterpret_tensor(buf202, (512, 128, 784), (100352, 784, 1), 0); del buf202  # reuse
        buf644 = reinterpret_tensor(buf194, (512, 128, 784), (100352, 784, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf628, buf633, buf640, buf644, 65536, 784, stream=stream0)
        del buf628
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf635, (512, 128, 784), (100352, 784, 1), 0), buf640, buf638, buf639, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf643 = buf640; del buf640  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf635, (512, 128, 784), (100352, 784, 1), 0), buf644, rsqrt_22, primals_136, buf638, buf639, buf643, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_136
        del rsqrt_22
        buf646 = reinterpret_tensor(buf644, (100352, 512), (512, 1), 0); del buf644  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_282, getitem_283, getitem_284, buf646, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_282
        del getitem_283
        del getitem_284
        buf648 = buf629; del buf629  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf649 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf643, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf648, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_23, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_23
        buf650 = buf649[0]
        assert_size_stride(buf650, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf650, 16, 'torch.ops.aten.convolution_backward.default')
        del buf649
        buf651 = buf648; del buf648  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf652 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf643, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf646, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf651, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf653 = buf652[1]
        assert_size_stride(buf653, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf653, 16, 'torch.ops.aten.convolution_backward.default')
        del buf652
        buf654 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(buf653, buf654, 147456, stream=stream0)
        del buf653
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_281, reinterpret_tensor(buf655, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_281
        buf657 = buf646; del buf646  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_276, getitem_277, getitem_278, buf657, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_276
        del getitem_277
        del getitem_278
        buf661 = buf643; del buf643  # reuse
        buf665 = reinterpret_tensor(buf635, (512, 128, 784), (100352, 784, 1), 0); del buf635  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf650, buf655, buf661, buf665, 65536, 784, stream=stream0)
        del buf650
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf657, (512, 128, 784), (100352, 784, 1), 0), buf661, buf659, buf660, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf664 = buf661; del buf661  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf657, (512, 128, 784), (100352, 784, 1), 0), buf665, rsqrt_21, primals_130, buf659, buf660, buf664, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_130
        del rsqrt_21
        buf667 = reinterpret_tensor(buf621, (401408, 512), (512, 1), 0); del buf621  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_269, getitem_270, getitem_271, buf667, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_269
        del getitem_270
        del getitem_271
        buf669 = buf651; del buf651  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf670 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf664, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf669, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_22, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_22
        buf671 = buf670[0]
        assert_size_stride(buf671, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf671, 16, 'torch.ops.aten.convolution_backward.default')
        del buf670
        buf672 = buf669; del buf669  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf673 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf664, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf667, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf672, (128, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf674 = buf673[1]
        assert_size_stride(buf674, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf674, 16, 'torch.ops.aten.convolution_backward.default')
        del buf673
        buf675 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_45.run(buf674, buf675, 65536, stream=stream0)
        del buf674
        buf676 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        buf738 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_50.run(buf611, buf676, buf738, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_268, reinterpret_tensor(buf676, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_268
        buf678 = reinterpret_tensor(buf667, (512, 512, 28, 28), (401408, 784, 28, 1), 0); del buf667  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_51.run(buf547, buf606, buf612, buf671, buf676, buf678, 262144, 784, stream=stream0)
        buf679 = reinterpret_tensor(buf671, (401408, 512), (512, 1), 0); del buf671  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_263, getitem_264, getitem_265, buf679, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_263
        del getitem_264
        del getitem_265
        buf681 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf682 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf742 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf743 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(buf34, buf681, buf682, buf742, buf743, 512, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf679, (512, 512, 784), (401408, 784, 1), 0), reinterpret_tensor(buf678, (512, 512, 784), (401408, 784, 1), 0), buf681, buf682, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf685 = reinterpret_tensor(buf606, (512, 512, 784), (401408, 784, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf679, (512, 512, 784), (401408, 784, 1), 0), reinterpret_tensor(buf678, (512, 512, 784), (401408, 784, 1), 0), rsqrt_20, primals_124, buf681, buf682, buf685, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_124
        del rsqrt_20
        buf687 = reinterpret_tensor(buf664, (100352, 512), (512, 1), 0); del buf664  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_256, getitem_257, getitem_258, buf687, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_256
        del getitem_257
        del getitem_258
        buf689 = buf672; del buf672  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf690 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf685, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf689, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_21
        buf691 = buf690[0]
        assert_size_stride(buf691, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf691, 16, 'torch.ops.aten.convolution_backward.default')
        del buf690
        buf692 = buf689; del buf689  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf693 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf685, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf687, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf692, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf694 = buf693[1]
        assert_size_stride(buf694, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf694, 16, 'torch.ops.aten.convolution_backward.default')
        del buf693
        buf695 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_45.run(buf694, buf695, 65536, stream=stream0)
        del buf694
        buf696 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf717 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf6, buf696, buf717, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_255, reinterpret_tensor(buf696, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_255
        buf698 = buf687; del buf687  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_250, getitem_251, getitem_252, buf698, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_250
        del getitem_251
        del getitem_252
        buf700 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf701 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf721 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf722 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf637, buf700, buf701, buf721, buf722, 128, stream=stream0)
        buf702 = buf665; del buf665  # reuse
        buf706 = reinterpret_tensor(buf657, (512, 128, 784), (100352, 784, 1), 0); del buf657  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf691, buf696, buf702, buf706, 65536, 784, stream=stream0)
        del buf691
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf698, (512, 128, 784), (100352, 784, 1), 0), buf702, buf700, buf701, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf705 = buf702; del buf702  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf698, (512, 128, 784), (100352, 784, 1), 0), buf706, rsqrt_19, primals_118, buf700, buf701, buf705, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_118
        del rsqrt_19
        buf708 = reinterpret_tensor(buf706, (100352, 512), (512, 1), 0); del buf706  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_243, getitem_244, getitem_245, buf708, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_243
        del getitem_244
        del getitem_245
        buf710 = buf692; del buf692  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf711 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf705, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf710, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_20, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_20
        buf712 = buf711[0]
        assert_size_stride(buf712, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf712, 16, 'torch.ops.aten.convolution_backward.default')
        del buf711
        buf713 = buf710; del buf710  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf714 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf705, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf708, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf713, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf715 = buf714[1]
        assert_size_stride(buf715, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf715, 16, 'torch.ops.aten.convolution_backward.default')
        del buf714
        buf716 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(buf715, buf716, 147456, stream=stream0)
        del buf715
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_242, reinterpret_tensor(buf717, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_242
        buf719 = buf708; del buf708  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_237, getitem_238, getitem_239, buf719, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_237
        del getitem_238
        del getitem_239
        buf723 = buf705; del buf705  # reuse
        buf727 = reinterpret_tensor(buf698, (512, 128, 784), (100352, 784, 1), 0); del buf698  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf712, buf717, buf723, buf727, 65536, 784, stream=stream0)
        del buf712
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf719, (512, 128, 784), (100352, 784, 1), 0), buf723, buf721, buf722, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf726 = buf723; del buf723  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf719, (512, 128, 784), (100352, 784, 1), 0), buf727, rsqrt_18, primals_112, buf721, buf722, buf726, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_112
        del rsqrt_18
        buf729 = reinterpret_tensor(buf685, (401408, 512), (512, 1), 0); del buf685  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_230, getitem_231, getitem_232, buf729, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_230
        del getitem_231
        del getitem_232
        buf731 = buf713; del buf713  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf732 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf726, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf731, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_19
        buf733 = buf732[0]
        assert_size_stride(buf733, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf733, 16, 'torch.ops.aten.convolution_backward.default')
        del buf732
        buf734 = buf731; del buf731  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf735 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf726, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf729, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf734, (128, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf736 = buf735[1]
        assert_size_stride(buf736, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf736, 16, 'torch.ops.aten.convolution_backward.default')
        del buf735
        buf737 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_45.run(buf736, buf737, 65536, stream=stream0)
        del buf736
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_229, reinterpret_tensor(buf738, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_229
        buf740 = buf729; del buf729  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_224, getitem_225, getitem_226, buf740, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_224
        del getitem_225
        del getitem_226
        buf744 = reinterpret_tensor(buf679, (512, 512, 784), (401408, 784, 1), 0); del buf679  # reuse
        buf748 = reinterpret_tensor(buf547, (512, 512, 784), (401408, 784, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_52.run(buf678, buf733, buf738, buf744, buf748, 262144, 784, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf740, (512, 512, 784), (401408, 784, 1), 0), buf744, buf742, buf743, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf747 = buf744; del buf744  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf740, (512, 512, 784), (401408, 784, 1), 0), buf748, rsqrt_17, primals_106, buf742, buf743, buf747, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_106
        del rsqrt_17
        buf750 = reinterpret_tensor(buf726, (100352, 512), (512, 1), 0); del buf726  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_217, getitem_218, getitem_219, buf750, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_217
        del getitem_218
        del getitem_219
        buf752 = buf734; del buf734  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf753 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf747, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf752, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_18
        buf754 = buf753[0]
        assert_size_stride(buf754, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf754, 16, 'torch.ops.aten.convolution_backward.default')
        del buf753
        buf755 = buf752; del buf752  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf756 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf747, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf750, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf755, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf757 = buf756[1]
        assert_size_stride(buf757, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf757, 16, 'torch.ops.aten.convolution_backward.default')
        del buf756
        buf758 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_45.run(buf757, buf758, 65536, stream=stream0)
        del buf757
        buf759 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        buf780 = empty_strided_cuda((51380224, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf6, buf759, buf780, 51380224, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_216, reinterpret_tensor(buf759, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_216
        buf761 = buf750; del buf750  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_211, getitem_212, getitem_213, buf761, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_211
        del getitem_212
        del getitem_213
        buf763 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf764 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf784 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf785 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_47.run(buf637, buf763, buf764, buf784, buf785, 128, stream=stream0)
        buf765 = buf727; del buf727  # reuse
        buf769 = reinterpret_tensor(buf719, (512, 128, 784), (100352, 784, 1), 0); del buf719  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf754, buf759, buf765, buf769, 65536, 784, stream=stream0)
        del buf754
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf761, (512, 128, 784), (100352, 784, 1), 0), buf765, buf763, buf764, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf768 = buf765; del buf765  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf761, (512, 128, 784), (100352, 784, 1), 0), buf769, rsqrt_16, primals_100, buf763, buf764, buf768, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_100
        del rsqrt_16
        buf771 = reinterpret_tensor(buf769, (100352, 512), (512, 1), 0); del buf769  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_204, getitem_205, getitem_206, buf771, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_204
        del getitem_205
        del getitem_206
        buf773 = buf755; del buf755  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf774 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf768, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf773, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_17, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_17
        buf775 = buf774[0]
        assert_size_stride(buf775, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf775, 16, 'torch.ops.aten.convolution_backward.default')
        del buf774
        buf776 = buf773; del buf773  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf777 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf768, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf771, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf776, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf778 = buf777[1]
        assert_size_stride(buf778, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf778, 16, 'torch.ops.aten.convolution_backward.default')
        del buf777
        buf779 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(buf778, buf779, 147456, stream=stream0)
        del buf778
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_203, reinterpret_tensor(buf780, (100352, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del getitem_203
        buf782 = buf771; del buf771  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_198, getitem_199, getitem_200, buf782, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_198
        del getitem_199
        del getitem_200
        buf786 = buf768; del buf768  # reuse
        buf790 = reinterpret_tensor(buf761, (512, 128, 784), (100352, 784, 1), 0); del buf761  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf775, buf780, buf786, buf790, 65536, 784, stream=stream0)
        del buf775
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf782, (512, 128, 784), (100352, 784, 1), 0), buf786, buf784, buf785, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf789 = buf786; del buf786  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf782, (512, 128, 784), (100352, 784, 1), 0), buf790, rsqrt_15, primals_94, buf784, buf785, buf789, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del primals_94
        del rsqrt_15
        buf792 = reinterpret_tensor(buf747, (401408, 512), (512, 1), 0); del buf747  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_191, getitem_192, getitem_193, buf792, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_191
        del getitem_192
        del getitem_193
        buf794 = buf776; del buf776  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf795 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf789, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf794, (512, 512, 28, 28), (0, 0, 0, 0), 0), convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_16
        buf796 = buf795[0]
        assert_size_stride(buf796, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf796, 16, 'torch.ops.aten.convolution_backward.default')
        del buf795
        buf797 = buf794; del buf794  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf798 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf789, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf792, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf797, (128, 512, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf799 = buf798[1]
        assert_size_stride(buf799, (128, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf799, 16, 'torch.ops.aten.convolution_backward.default')
        del buf798
        buf800 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_45.run(buf799, buf800, 65536, stream=stream0)
        del buf799
        buf801 = empty_strided_cuda((205520896, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_42.run(buf611, buf801, 205520896, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_190, reinterpret_tensor(buf801, (401408, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del getitem_190
        buf803 = buf792; del buf792  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_185, getitem_186, getitem_187, buf803, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_185
        del getitem_186
        del getitem_187
        buf805 = buf733; del buf733  # reuse
        buf808 = buf748; del buf748  # reuse
        buf812 = reinterpret_tensor(buf740, (512, 512, 784), (401408, 784, 1), 0); del buf740  # reuse
        buf826 = buf622; del buf622  # reuse
        buf830 = reinterpret_tensor(buf614, (512, 512, 784), (401408, 784, 1), 0); del buf614  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_53.run(buf805, buf678, buf738, buf796, buf801, buf808, buf812, buf826, buf830, 262144, 784, stream=stream0)
        del buf678
        del buf796
        del buf805
        buf806 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf807 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf825 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_54.run(buf34, buf806, buf807, buf825, 512, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf803, (512, 512, 784), (401408, 784, 1), 0), buf808, buf806, buf807, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf811 = buf808; del buf808  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf803, (512, 512, 784), (401408, 784, 1), 0), buf812, rsqrt_14, primals_88, buf806, buf807, buf811, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del buf803
        del buf812
        del primals_88
        del rsqrt_14
        buf814 = empty_strided_cuda((802816, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_178, getitem_179, getitem_180, buf814, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_178
        del getitem_179
        del getitem_180
        buf816 = buf797; del buf797  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf817 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf811, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf816, (512, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_15, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_15
        buf818 = buf817[0]
        assert_size_stride(buf818, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf818, 16, 'torch.ops.aten.convolution_backward.default')
        del buf817
        buf819 = buf816; del buf816  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf820 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf811, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf814, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf819, (512, 256, 1, 1), (0, 0, 0, 0), 0), None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf821 = buf820[1]
        assert_size_stride(buf821, (512, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf821, 16, 'torch.ops.aten.convolution_backward.default')
        del buf820
        buf822 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(buf821, buf822, 131072, stream=stream0)
        del buf821
        buf823 = reinterpret_tensor(buf811, (401408, 512), (512, 1), 0); del buf811  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_175, getitem_176, getitem_177, buf823, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_175
        del getitem_176
        del getitem_177
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_14.run(reinterpret_tensor(buf823, (512, 512, 784), (401408, 784, 1), 0), buf826, buf34, buf825, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        buf829 = buf826; del buf826  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_15.run(reinterpret_tensor(buf823, (512, 512, 784), (401408, 784, 1), 0), buf830, rsqrt_13, primals_82, buf34, buf825, buf829, 401408, 784, 401408, 784, 1024, 512, 392, 1, stream=stream0)
        del primals_82
        del rsqrt_13
        buf832 = reinterpret_tensor(buf789, (100352, 512), (512, 1), 0); del buf789  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_168, getitem_169, getitem_170, buf832, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_168
        del getitem_169
        del getitem_170
        buf834 = buf819; del buf819  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf835 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf829, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf834, (512, 128, 28, 28), (0, 0, 0, 0), 0), convert_element_type_14, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_14
        buf836 = buf835[0]
        assert_size_stride(buf836, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf836, 16, 'torch.ops.aten.convolution_backward.default')
        del buf835
        buf837 = buf834; del buf834  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf838 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf829, (512, 512, 28, 28), (401408, 784, 28, 1), 0), reinterpret_tensor(buf832, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf837, (512, 128, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf839 = buf838[1]
        assert_size_stride(buf839, (512, 128, 1, 1), (128, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf839, 16, 'torch.ops.aten.convolution_backward.default')
        del buf838
        buf840 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_45.run(buf839, buf840, 65536, stream=stream0)
        del buf839
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_167, buf6, 16, 1, 512, 1, 1, 32, 16, 100352, 1, 1, stream=stream0)
        del buf136
        del buf192
        del buf633
        del buf655
        del buf696
        del buf7
        del buf717
        del buf73
        del buf759
        del buf780
        del getitem_167
        buf842 = buf832; del buf832  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_162, getitem_163, getitem_164, buf842, 32, 1, 512, 1, 2, 16, 32, 100352, 1, 1, stream=stream0)
        del getitem_162
        del getitem_163
        del getitem_164
        buf844 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf845 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf864 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_55.run(buf637, buf844, buf845, buf864, 128, stream=stream0)
        buf846 = buf790; del buf790  # reuse
        buf850 = reinterpret_tensor(buf782, (512, 128, 784), (100352, 784, 1), 0); del buf782  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_48.run(buf836, buf6, buf846, buf850, 65536, 784, stream=stream0)
        del buf6
        del buf836
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_16.run(reinterpret_tensor(buf842, (512, 128, 784), (100352, 784, 1), 0), buf846, buf844, buf845, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        buf849 = buf846; del buf846  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_17.run(reinterpret_tensor(buf842, (512, 128, 784), (100352, 784, 1), 0), buf850, rsqrt_12, primals_76, buf844, buf845, buf849, 401408, 784, 100352, 784, 1024, 128, 392, 1, stream=stream0)
        del buf842
        del buf850
        del primals_76
        del rsqrt_12
        buf852 = reinterpret_tensor(buf829, (401408, 512), (512, 1), 0); del buf829  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_155, getitem_156, getitem_157, buf852, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_155
        del getitem_156
        del getitem_157
        buf854 = buf837; del buf837  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf855 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf849, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf854, (512, 128, 56, 56), (0, 0, 0, 0), 0), convert_element_type_13, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_13
        buf856 = buf855[0]
        assert_size_stride(buf856, (512, 128, 56, 56), (401408, 1, 7168, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf856, 16, 'torch.ops.aten.convolution_backward.default')
        del buf855
        buf857 = buf854; del buf854  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf858 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf849, (512, 128, 28, 28), (100352, 784, 28, 1), 0), reinterpret_tensor(buf852, (512, 128, 56, 56), (401408, 3136, 56, 1), 0), reinterpret_tensor(buf857, (128, 128, 3, 3), (0, 0, 0, 0), 0), None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        del buf849
        buf859 = buf858[1]
        assert_size_stride(buf859, (128, 128, 3, 3), (1152, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf859, 16, 'torch.ops.aten.convolution_backward.default')
        del buf858
        buf860 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_49.run(buf859, buf860, 147456, stream=stream0)
        del buf859
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_154, buf611, 16, 1, 512, 1, 1, 32, 16, 401408, 1, 1, stream=stream0)
        del buf612
        del buf676
        del buf738
        del buf801
        del getitem_154
        buf862 = buf852; del buf852  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_149, getitem_150, getitem_151, buf862, 32, 1, 512, 1, 2, 16, 32, 401408, 1, 1, stream=stream0)
        del getitem_149
        del getitem_150
        del getitem_151
        buf865 = reinterpret_tensor(buf830, (512, 128, 3136), (401408, 3136, 1), 0); del buf830  # reuse
        buf869 = reinterpret_tensor(buf823, (512, 128, 3136), (401408, 3136, 1), 0); del buf823  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_56.run(buf856, buf611, buf865, buf869, 65536, 3136, stream=stream0)
        del buf611
        del buf856
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_18.run(reinterpret_tensor(buf862, (512, 128, 3136), (401408, 3136, 1), 0), buf865, buf637, buf864, 1605632, 3136, 401408, 3136, 1024, 128, 1568, 1, stream=stream0)
        buf868 = buf865; del buf865  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_19.run(reinterpret_tensor(buf862, (512, 128, 3136), (401408, 3136, 1), 0), buf869, rsqrt_11, primals_70, buf637, buf864, buf868, 1605632, 3136, 401408, 3136, 1024, 128, 1568, 1, stream=stream0)
        del buf862
        del buf869
        del primals_70
        del rsqrt_11
        buf871 = buf814; del buf814  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_142, getitem_143, getitem_144, buf871, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_142
        del getitem_143
        del getitem_144
        buf873 = buf857; del buf857  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf874 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf868, (512, 128, 56, 56), (401408, 3136, 56, 1), 0), reinterpret_tensor(buf873, (512, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_12
        buf875 = buf874[0]
        assert_size_stride(buf875, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf875, 16, 'torch.ops.aten.convolution_backward.default')
        del buf874
        buf876 = buf873; del buf873  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf877 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf868, (512, 128, 56, 56), (401408, 3136, 56, 1), 0), reinterpret_tensor(buf871, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf876, (128, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        del buf868
        buf878 = buf877[1]
        assert_size_stride(buf878, (128, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf878, 16, 'torch.ops.aten.convolution_backward.default')
        del buf877
        buf879 = empty_strided_cuda((128, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_57.run(buf878, buf879, 32768, stream=stream0)
        del buf878
        buf880 = empty_strided_cuda((802816, 512), (512, 1), torch.int8)
        # Topologically Sorted Source Nodes: [full_380], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_58.run(buf880, 411041792, stream=stream0)
        buf881 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_59.run(buf880, buf881, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_141, reinterpret_tensor(buf881, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_141
        buf883 = buf871; del buf871  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_136, getitem_137, getitem_138, buf883, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_136
        del getitem_137
        del getitem_138
        buf885 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf886 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf950 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf951 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_29.run(buf241, buf885, buf886, buf950, buf951, 256, stream=stream0)
        buf887 = empty_strided_cuda((512, 256, 3136), (802816, 3136, 1), torch.bfloat16)
        buf891 = empty_strided_cuda((512, 256, 3136), (802816, 3136, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_60.run(buf818, buf875, buf881, buf887, buf891, 131072, 3136, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf883, (512, 256, 3136), (802816, 3136, 1), 0), buf887, buf885, buf886, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf890 = buf887; del buf887  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf883, (512, 256, 3136), (802816, 3136, 1), 0), buf891, rsqrt_10, primals_64, buf885, buf886, buf890, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_64
        del rsqrt_10
        buf893 = reinterpret_tensor(buf599, (200704, 512), (512, 1), 0); del buf599  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_129, getitem_130, getitem_131, buf893, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_129
        del getitem_130
        del getitem_131
        buf895 = buf876; del buf876  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf896 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf890, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf895, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_11, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_11
        buf897 = buf896[0]
        assert_size_stride(buf897, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf897, 16, 'torch.ops.aten.convolution_backward.default')
        del buf896
        buf898 = buf895; del buf895  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf899 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf890, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf893, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf898, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf900 = buf899[1]
        assert_size_stride(buf900, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf900, 16, 'torch.ops.aten.convolution_backward.default')
        del buf899
        buf901 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_61.run(buf900, buf901, 16384, stream=stream0)
        del buf900
        buf902 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf924 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf213, buf902, buf924, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_128, reinterpret_tensor(buf902, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_128
        buf904 = buf893; del buf893  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_123, getitem_124, getitem_125, buf904, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_123
        del getitem_124
        del getitem_125
        buf906 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [bn1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_62.run(buf906, 64, stream=stream0)
        buf907 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf908 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf928 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf929 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_63.run(buf906, buf907, buf908, buf928, buf929, 64, stream=stream0)
        buf909 = reinterpret_tensor(buf600, (512, 64, 3136), (200704, 3136, 1), 0); del buf600  # reuse
        buf913 = reinterpret_tensor(buf592, (512, 64, 3136), (200704, 3136, 1), 0); del buf592  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_64.run(buf897, buf902, buf909, buf913, 32768, 3136, stream=stream0)
        del buf897
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf904, (512, 64, 3136), (200704, 3136, 1), 0), buf909, buf907, buf908, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf912 = buf909; del buf909  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf904, (512, 64, 3136), (200704, 3136, 1), 0), buf913, rsqrt_9, primals_58, buf907, buf908, buf912, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_58
        del rsqrt_9
        buf915 = reinterpret_tensor(buf913, (200704, 512), (512, 1), 0); del buf913  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_116, getitem_117, getitem_118, buf915, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_116
        del getitem_117
        del getitem_118
        buf917 = buf898; del buf898  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf918 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf912, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf917, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_10, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_10
        buf919 = buf918[0]
        assert_size_stride(buf919, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf919, 16, 'torch.ops.aten.convolution_backward.default')
        del buf918
        buf920 = buf917; del buf917  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf921 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf912, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf915, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf920, (64, 64, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf922 = buf921[1]
        assert_size_stride(buf922, (64, 64, 3, 3), (576, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf922, 16, 'torch.ops.aten.convolution_backward.default')
        del buf921
        buf923 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_65.run(buf922, buf923, 36864, stream=stream0)
        del buf922
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_115, reinterpret_tensor(buf924, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_115
        buf926 = buf915; del buf915  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_110, getitem_111, getitem_112, buf926, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_110
        del getitem_111
        del getitem_112
        buf930 = buf912; del buf912  # reuse
        buf934 = reinterpret_tensor(buf904, (512, 64, 3136), (200704, 3136, 1), 0); del buf904  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_64.run(buf919, buf924, buf930, buf934, 32768, 3136, stream=stream0)
        del buf919
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf926, (512, 64, 3136), (200704, 3136, 1), 0), buf930, buf928, buf929, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf933 = buf930; del buf930  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf926, (512, 64, 3136), (200704, 3136, 1), 0), buf934, rsqrt_8, primals_52, buf928, buf929, buf933, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_52
        del rsqrt_8
        buf936 = reinterpret_tensor(buf890, (802816, 512), (512, 1), 0); del buf890  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_103, getitem_104, getitem_105, buf936, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_103
        del getitem_104
        del getitem_105
        buf938 = buf920; del buf920  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf939 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf933, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf938, (512, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_9
        buf940 = buf939[0]
        assert_size_stride(buf940, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf940, 16, 'torch.ops.aten.convolution_backward.default')
        del buf939
        buf941 = buf938; del buf938  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf942 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf933, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf936, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf941, (64, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf943 = buf942[1]
        assert_size_stride(buf943, (64, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf943, 16, 'torch.ops.aten.convolution_backward.default')
        del buf942
        buf944 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_61.run(buf943, buf944, 16384, stream=stream0)
        del buf943
        buf945 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        buf1007 = empty_strided_cuda((411041792, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_66.run(buf880, buf945, buf1007, 411041792, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_102, reinterpret_tensor(buf945, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_102
        buf947 = reinterpret_tensor(buf936, (512, 256, 56, 56), (802816, 3136, 56, 1), 0); del buf936  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_67.run(buf818, buf875, buf881, buf940, buf945, buf947, 131072, 3136, stream=stream0)
        buf948 = reinterpret_tensor(buf940, (802816, 512), (512, 1), 0); del buf940  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_97, getitem_98, getitem_99, buf948, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_97
        del getitem_98
        del getitem_99
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf948, (512, 256, 3136), (802816, 3136, 1), 0), reinterpret_tensor(buf947, (512, 256, 3136), (802816, 3136, 1), 0), buf950, buf951, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf954 = reinterpret_tensor(buf875, (512, 256, 3136), (802816, 3136, 1), 0); del buf875  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf948, (512, 256, 3136), (802816, 3136, 1), 0), reinterpret_tensor(buf947, (512, 256, 3136), (802816, 3136, 1), 0), rsqrt_7, primals_46, buf950, buf951, buf954, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_46
        del rsqrt_7
        buf956 = reinterpret_tensor(buf933, (200704, 512), (512, 1), 0); del buf933  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_90, getitem_91, getitem_92, buf956, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_90
        del getitem_91
        del getitem_92
        buf958 = buf941; del buf941  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf959 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf954, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf958, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_8, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_8
        buf960 = buf959[0]
        assert_size_stride(buf960, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf960, 16, 'torch.ops.aten.convolution_backward.default')
        del buf959
        buf961 = buf958; del buf958  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf962 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf954, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf956, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf961, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf963 = buf962[1]
        assert_size_stride(buf963, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf963, 16, 'torch.ops.aten.convolution_backward.default')
        del buf962
        buf964 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_61.run(buf963, buf964, 16384, stream=stream0)
        del buf963
        buf965 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        buf986 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_32.run(buf213, buf965, buf986, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_89, reinterpret_tensor(buf965, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_89
        buf967 = buf956; del buf956  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_84, getitem_85, getitem_86, buf967, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_84
        del getitem_85
        del getitem_86
        buf969 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf970 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf990 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf991 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_63.run(buf906, buf969, buf970, buf990, buf991, 64, stream=stream0)
        buf971 = buf934; del buf934  # reuse
        buf975 = reinterpret_tensor(buf926, (512, 64, 3136), (200704, 3136, 1), 0); del buf926  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_64.run(buf960, buf965, buf971, buf975, 32768, 3136, stream=stream0)
        del buf960
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf967, (512, 64, 3136), (200704, 3136, 1), 0), buf971, buf969, buf970, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf974 = buf971; del buf971  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf967, (512, 64, 3136), (200704, 3136, 1), 0), buf975, rsqrt_6, primals_40, buf969, buf970, buf974, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_40
        del rsqrt_6
        buf977 = reinterpret_tensor(buf975, (200704, 512), (512, 1), 0); del buf975  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_77, getitem_78, getitem_79, buf977, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_77
        del getitem_78
        del getitem_79
        buf979 = buf961; del buf961  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf980 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf974, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf979, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_7
        buf981 = buf980[0]
        assert_size_stride(buf981, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf981, 16, 'torch.ops.aten.convolution_backward.default')
        del buf980
        buf982 = buf979; del buf979  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf983 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf974, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf977, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf982, (64, 64, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf984 = buf983[1]
        assert_size_stride(buf984, (64, 64, 3, 3), (576, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf984, 16, 'torch.ops.aten.convolution_backward.default')
        del buf983
        buf985 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_65.run(buf984, buf985, 36864, stream=stream0)
        del buf984
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_76, reinterpret_tensor(buf986, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_76
        buf988 = buf977; del buf977  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_71, getitem_72, getitem_73, buf988, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_71
        del getitem_72
        del getitem_73
        buf992 = buf974; del buf974  # reuse
        buf996 = reinterpret_tensor(buf967, (512, 64, 3136), (200704, 3136, 1), 0); del buf967  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_64.run(buf981, buf986, buf992, buf996, 32768, 3136, stream=stream0)
        del buf981
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf988, (512, 64, 3136), (200704, 3136, 1), 0), buf992, buf990, buf991, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf995 = buf992; del buf992  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf988, (512, 64, 3136), (200704, 3136, 1), 0), buf996, rsqrt_5, primals_34, buf990, buf991, buf995, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_34
        del rsqrt_5
        buf998 = reinterpret_tensor(buf954, (802816, 512), (512, 1), 0); del buf954  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_64, getitem_65, getitem_66, buf998, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_64
        del getitem_65
        del getitem_66
        buf1000 = buf982; del buf982  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1001 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf995, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1000, (512, 256, 56, 56), (0, 0, 0, 0), 0), convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_6
        buf1002 = buf1001[0]
        assert_size_stride(buf1002, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1002, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1001
        buf1003 = buf1000; del buf1000  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1004 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf995, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf998, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1003, (64, 256, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1005 = buf1004[1]
        assert_size_stride(buf1005, (64, 256, 1, 1), (256, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1005, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1004
        buf1006 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_61.run(buf1005, buf1006, 16384, stream=stream0)
        del buf1005
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_63, reinterpret_tensor(buf1007, (802816, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del getitem_63
        buf1009 = buf998; del buf998  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_58, getitem_59, getitem_60, buf1009, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_58
        del getitem_59
        del getitem_60
        buf1011 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1012 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf1030 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_68.run(buf241, buf1011, buf1012, buf1030, 256, stream=stream0)
        buf1013 = reinterpret_tensor(buf948, (512, 256, 3136), (802816, 3136, 1), 0); del buf948  # reuse
        buf1017 = reinterpret_tensor(buf818, (512, 256, 3136), (802816, 3136, 1), 0); del buf818  # reuse
        buf1031 = buf891; del buf891  # reuse
        buf1035 = reinterpret_tensor(buf883, (512, 256, 3136), (802816, 3136, 1), 0); del buf883  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_69.run(buf947, buf1002, buf1007, buf1013, buf1017, buf1031, buf1035, 131072, 3136, stream=stream0)
        del buf1002
        del buf947
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf1009, (512, 256, 3136), (802816, 3136, 1), 0), buf1013, buf1011, buf1012, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf1016 = buf1013; del buf1013  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf1009, (512, 256, 3136), (802816, 3136, 1), 0), buf1017, rsqrt_4, primals_28, buf1011, buf1012, buf1016, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del buf1009
        del primals_28
        del rsqrt_4
        buf1019 = reinterpret_tensor(buf995, (200704, 512), (512, 1), 0); del buf995  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_51, getitem_52, getitem_53, buf1019, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_51
        del getitem_52
        del getitem_53
        buf1021 = buf1003; del buf1003  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1022 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1016, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1021, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_5
        buf1023 = buf1022[0]
        assert_size_stride(buf1023, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1023, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1022
        buf1024 = buf1021; del buf1021  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1025 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1016, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1019, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1024, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1026 = buf1025[1]
        assert_size_stride(buf1026, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1026, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1025
        buf1027 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_61.run(buf1026, buf1027, 16384, stream=stream0)
        del buf1026
        buf1028 = reinterpret_tensor(buf1016, (802816, 512), (512, 1), 0); del buf1016  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_48, getitem_49, getitem_50, buf1028, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_48
        del getitem_49
        del getitem_50
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_20.run(reinterpret_tensor(buf1028, (512, 256, 3136), (802816, 3136, 1), 0), buf1031, buf241, buf1030, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        buf1034 = buf1031; del buf1031  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_21.run(reinterpret_tensor(buf1028, (512, 256, 3136), (802816, 3136, 1), 0), buf1035, rsqrt_3, primals_22, buf241, buf1030, buf1034, 1605632, 3136, 802816, 3136, 1024, 256, 1568, 1, stream=stream0)
        del primals_22
        del rsqrt_3
        buf1037 = buf1019; del buf1019  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_41, getitem_42, getitem_43, buf1037, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_41
        del getitem_42
        del getitem_43
        buf1039 = buf1024; del buf1024  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1040 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1034, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1039, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_4, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_4
        buf1041 = buf1040[0]
        assert_size_stride(buf1041, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1041, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1040
        buf1042 = buf1039; del buf1039  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1043 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1034, (512, 256, 56, 56), (802816, 3136, 56, 1), 0), reinterpret_tensor(buf1037, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1042, (256, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        buf1044 = buf1043[1]
        assert_size_stride(buf1044, (256, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1044, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1043
        buf1045 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_61.run(buf1044, buf1045, 16384, stream=stream0)
        del buf1044
        buf1046 = empty_strided_cuda((102760448, ), (1, ), torch.int8)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_21.run(buf213, buf1046, 102760448, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_40, reinterpret_tensor(buf1046, (200704, 512), (512, 1), 0), 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del getitem_40
        buf1048 = buf1037; del buf1037  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_35, getitem_36, getitem_37, buf1048, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_35
        del getitem_36
        del getitem_37
        buf1050 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1051 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1070 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1071 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf1092 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_70.run(buf906, buf1050, buf1051, buf1070, buf1071, buf1092, 64, stream=stream0)
        buf1052 = buf996; del buf996  # reuse
        buf1056 = reinterpret_tensor(buf988, (512, 64, 3136), (200704, 3136, 1), 0); del buf988  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_64.run(buf1041, buf1046, buf1052, buf1056, 32768, 3136, stream=stream0)
        del buf1041
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf1048, (512, 64, 3136), (200704, 3136, 1), 0), buf1052, buf1050, buf1051, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf1055 = buf1052; del buf1052  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf1048, (512, 64, 3136), (200704, 3136, 1), 0), buf1056, rsqrt_2, primals_16, buf1050, buf1051, buf1055, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del primals_16
        del rsqrt_2
        buf1058 = reinterpret_tensor(buf1056, (200704, 512), (512, 1), 0); del buf1056  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_28, getitem_29, getitem_30, buf1058, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_28
        del getitem_29
        del getitem_30
        buf1060 = buf1042; del buf1042  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1061 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1055, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1060, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_3, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_3
        buf1062 = buf1061[0]
        assert_size_stride(buf1062, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1062, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1061
        buf1063 = buf1060; del buf1060  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1064 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1055, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1058, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1063, (64, 64, 3, 3), (0, 0, 0, 0), 0), None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False])
        buf1065 = buf1064[1]
        assert_size_stride(buf1065, (64, 64, 3, 3), (576, 9, 3, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1065, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1064
        buf1066 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_65.run(buf1065, buf1066, 36864, stream=stream0)
        del buf1065
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_27, buf213, 16, 1, 512, 1, 1, 32, 16, 200704, 1, 1, stream=stream0)
        del buf1046
        del buf214
        del buf280
        del buf342
        del buf405
        del buf467
        del buf530
        del buf590
        del buf902
        del buf924
        del buf965
        del buf986
        del getitem_27
        buf1068 = buf1058; del buf1058  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_22, getitem_23, getitem_24, buf1068, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_22
        del getitem_23
        del getitem_24
        buf1072 = buf1055; del buf1055  # reuse
        buf1076 = reinterpret_tensor(buf1048, (512, 64, 3136), (200704, 3136, 1), 0); del buf1048  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_64.run(buf1062, buf213, buf1072, buf1076, 32768, 3136, stream=stream0)
        del buf1062
        del buf213
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_22.run(reinterpret_tensor(buf1068, (512, 64, 3136), (200704, 3136, 1), 0), buf1072, buf1070, buf1071, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        buf1075 = buf1072; del buf1072  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_23.run(reinterpret_tensor(buf1068, (512, 64, 3136), (200704, 3136, 1), 0), buf1076, rsqrt_1, primals_10, buf1070, buf1071, buf1075, 1605632, 3136, 200704, 3136, 1024, 64, 1568, 1, stream=stream0)
        del buf1068
        del primals_10
        del rsqrt_1
        buf1078 = reinterpret_tensor(buf1076, (200704, 512), (512, 1), 0); del buf1076  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_15, getitem_16, getitem_17, buf1078, 32, 1, 512, 1, 2, 16, 32, 200704, 1, 1, stream=stream0)
        del getitem_15
        del getitem_16
        del getitem_17
        buf1080 = buf1063; del buf1063  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1081 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1075, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1080, (512, 64, 56, 56), (0, 0, 0, 0), 0), convert_element_type_2, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False])
        del convert_element_type_2
        buf1082 = buf1081[0]
        assert_size_stride(buf1082, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1082, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1081
        buf1083 = buf1080; del buf1080  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1084 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1075, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1078, (512, 64, 56, 56), (200704, 3136, 56, 1), 0), reinterpret_tensor(buf1083, (64, 64, 1, 1), (0, 0, 0, 0), 0), None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False])
        del buf1075
        del buf1078
        buf1085 = buf1084[1]
        assert_size_stride(buf1085, (64, 64, 1, 1), (64, 1, 1, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1085, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1084
        buf1086 = empty_strided_cuda((64, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_71.run(buf1085, buf1086, 4096, stream=stream0)
        del buf1085
        buf1087 = buf1023; del buf1023  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_72.run(buf1087, buf1082, 102760448, stream=stream0)
        del buf1082
        buf1088 = reinterpret_tensor(buf1034, (512, 64, 112, 112), (802816, 12544, 112, 1), 0); del buf1034  # reuse
        # Topologically Sorted Source Nodes: [maxpool], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_73.run(getitem_14, buf1087, buf1088, 6422528, 64, stream=stream0)
        del buf1087
        del getitem_14
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        unpack_kernel_1.run(getitem_12, buf880, 16, 1, 512, 1, 1, 32, 16, 802816, 1, 1, stream=stream0)
        del buf1007
        del buf881
        del buf945
        del getitem_12
        buf1090 = reinterpret_tensor(buf1035, (802816, 512), (512, 1), 0); del buf1035  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem_7, getitem_8, getitem_9, buf1090, 32, 1, 512, 1, 2, 16, 32, 802816, 1, 1, stream=stream0)
        del getitem_7
        del getitem_8
        del getitem_9
        buf1093 = reinterpret_tensor(buf1028, (512, 64, 12544), (802816, 12544, 1), 0); del buf1028  # reuse
        buf1097 = reinterpret_tensor(buf1017, (512, 64, 12544), (802816, 12544, 1), 0); del buf1017  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_74.run(buf1088, buf880, buf1093, buf1097, 411041792, stream=stream0)
        del buf1088
        del buf880
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_reduce_kernel_24.run(reinterpret_tensor(buf1090, (512, 64, 12544), (802816, 12544, 1), 0), buf1093, buf906, buf1092, 6422528, 12544, 802816, 12544, 1024, 64, 6272, 1, stream=stream0)
        buf1096 = buf1093; del buf1093  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _bn_bwd_dx_kernel_25.run(reinterpret_tensor(buf1090, (512, 64, 12544), (802816, 12544, 1), 0), buf1097, rsqrt, primals_4, buf906, buf1092, buf1096, 6422528, 12544, 802816, 12544, 1024, 64, 6272, 1, stream=stream0)
        del buf1090
        del buf1097
        del primals_4
        del rsqrt
        buf1099 = empty_strided_cuda((150528, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        dequant_unpack_kernel_0.run(getitem, getitem_1, getitem_2, buf1099, 32, 1, 512, 1, 2, 16, 32, 150528, 1, 1, stream=stream0)
        del getitem
        del getitem_1
        del getitem_2
        buf1101 = buf1083; del buf1083  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf1102 = torch.ops.aten.convolution_backward.default(reinterpret_tensor(buf1096, (512, 64, 112, 112), (802816, 12544, 112, 1), 0), reinterpret_tensor(buf1099, (512, 3, 224, 224), (150528, 50176, 224, 1), 0), reinterpret_tensor(buf1101, (64, 3, 7, 7), (0, 0, 0, 0), 0), None, [2, 2], [3, 3], [1, 1], False, [0], 1, [False, True, False])
        del buf1096
        del buf1099
        del buf1101
        buf1103 = buf1102[1]
        assert_size_stride(buf1103, (64, 3, 7, 7), (147, 49, 7, 1), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf1103, 16, 'torch.ops.aten.convolution_backward.default')
        del buf1102
        buf1104 = empty_strided_cuda((64, 3, 7, 7), (147, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf1103, buf1104, 9408, stream=stream0)
        del buf1103
    return (buf1104, None, None, buf1092, buf906, None, None, buf1086, None, buf1071, buf1070, None, None, buf1066, None, buf1051, buf1050, None, None, buf1045, None, buf1030, buf241, None, None, buf1027, None, buf1012, buf1011, None, None, buf1006, None, buf991, buf990, None, None, buf985, None, buf970, buf969, None, None, buf964, None, buf951, buf950, None, None, buf944, None, buf929, buf928, None, None, buf923, None, buf908, buf907, None, None, buf901, None, buf886, buf885, None, None, buf879, None, buf864, buf637, None, None, buf860, None, buf845, buf844, None, None, buf840, None, buf825, buf34, None, None, buf822, None, buf807, buf806, None, None, buf800, None, buf785, buf784, None, None, buf779, None, buf764, buf763, None, None, buf758, None, buf743, buf742, None, None, buf737, None, buf722, buf721, None, None, buf716, None, buf701, buf700, None, None, buf695, None, buf682, buf681, None, None, buf675, None, buf660, buf659, None, None, buf654, None, buf639, buf638, None, None, buf632, None, buf617, buf616, None, None, buf610, None, buf595, buf594, None, None, buf589, None, buf574, buf573, None, None, buf569, None, buf554, buf218, None, None, buf551, None, buf536, buf535, None, None, buf529, None, buf514, buf513, None, None, buf508, None, buf493, buf492, None, None, buf487, None, buf472, buf471, None, None, buf466, None, buf451, buf450, None, None, buf445, None, buf430, buf429, None, None, buf424, None, buf411, buf410, None, None, buf404, None, buf389, buf388, None, None, buf383, None, buf368, buf367, None, None, buf362, None, buf347, buf346, None, None, buf341, None, buf326, buf325, None, None, buf320, None, buf305, buf304, None, None, buf299, None, buf286, buf285, None, None, buf279, None, buf264, buf263, None, None, buf258, None, buf243, buf242, None, None, buf235, None, buf220, buf219, None, None, buf212, None, buf197, buf196, None, None, buf191, None, buf176, buf175, None, None, buf171, None, buf158, buf11, None, None, buf155, None, buf142, buf141, None, None, buf135, None, buf120, buf119, None, None, buf114, None, buf99, buf98, None, None, buf93, None, buf78, buf77, None, None, buf72, None, buf57, buf56, None, None, buf51, None, buf36, buf35, None, None, buf28, None, buf13, buf12, None, None, buf5, )


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
    getitem = rand_strided((150528, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_1 = rand_strided((150528, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_2 = rand_strided((150528, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_8 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_10 = rand_strided((512, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_12 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_14 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.int8)
    convert_element_type_2 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_15 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_16 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_17 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_22 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_23 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_24 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_27 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_3 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_28 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_29 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_30 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_36 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_37 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_40 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_4 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_41 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_42 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_43 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_48 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_49 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_50 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_5 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_51 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_52 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_53 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_59 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_60 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_63 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_6 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_64 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_65 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_66 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_72 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_73 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_76 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_7 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_77 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_78 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_79 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_85 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_86 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_89 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_8 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_90 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_91 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_92 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_7 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_98 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_99 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_102 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_9 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_103 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_104 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_105 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_110 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_111 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_112 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_115 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_10 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_116 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_117 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_118 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_124 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_125 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_128 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_11 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_129 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_130 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_131 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_136 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_137 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_138 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_141 = rand_strided((802816, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_12 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_142 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_143 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_144 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_150 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_151 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_154 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_13 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_155 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_156 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_157 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_162 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_163 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_164 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_167 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_14 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_168 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_169 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_170 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_13 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_175 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_176 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_177 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_15 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_178 = rand_strided((802816, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_179 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_180 = rand_strided((802816, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_14 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_185 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_186 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_187 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_190 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_16 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_191 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_192 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_193 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_198 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_199 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_200 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_203 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_17 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_204 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_205 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_206 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_211 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_212 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_213 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_216 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_18 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_217 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_218 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_219 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_224 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_225 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_226 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_229 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_19 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_230 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_231 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_232 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_237 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_238 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_239 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_242 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_20 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_243 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_244 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_245 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_250 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_251 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_252 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_255 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_21 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_256 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_257 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_258 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_263 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_264 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_265 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_268 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_22 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_269 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_270 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_271 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_276 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_277 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_278 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_281 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_23 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_282 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_283 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_284 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_289 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_290 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_291 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_294 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_24 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    getitem_295 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_296 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_297 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_302 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_303 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_304 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_307 = rand_strided((401408, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_25 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_308 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_309 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_310 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_315 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_316 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_317 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_320 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_26 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_321 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_322 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_323 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_328 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_329 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_330 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_333 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_27 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_334 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_335 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_336 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_26 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_341 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_342 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_343 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_28 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_344 = rand_strided((401408, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_345 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_346 = rand_strided((401408, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_27 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_351 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_352 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_353 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_356 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_29 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_357 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_358 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_359 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_364 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_365 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_366 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_369 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_30 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_370 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_371 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_372 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_377 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_378 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_379 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_382 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_31 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_383 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_384 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_385 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_30 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_390 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_391 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_392 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_395 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_32 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_396 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_397 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_398 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_403 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_404 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_405 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_408 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_33 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_409 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_410 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_411 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_416 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_417 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_418 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_421 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_34 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_422 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_423 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_424 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_33 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_429 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_430 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_431 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_434 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_35 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_435 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_436 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_437 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_442 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_443 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_444 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_447 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_36 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_448 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_449 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_450 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_455 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_456 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_457 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_460 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_37 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_461 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_462 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_463 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_36 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_468 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_469 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_470 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_473 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_38 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_474 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_475 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_476 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_481 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_482 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_483 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_486 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_39 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_487 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_488 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_489 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_494 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_495 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_496 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_499 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_40 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_500 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_501 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_502 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_39 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_507 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_508 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_509 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_512 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_41 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_513 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_514 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_515 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_520 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_521 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_522 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_525 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_42 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_526 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_527 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_528 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_533 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_534 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_535 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_538 = rand_strided((50176, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_43 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    getitem_539 = rand_strided((50176, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_540 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_541 = rand_strided((50176, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_42 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_546 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_547 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_548 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_551 = rand_strided((200704, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_44 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_552 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_553 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_554 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_559 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_560 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_561 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_564 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_45 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_565 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_566 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_567 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_572 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_573 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_574 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_577 = rand_strided((25088, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_46 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_578 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_579 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_580 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_45 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_585 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_586 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_587 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_47 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    getitem_588 = rand_strided((200704, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_589 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_590 = rand_strided((200704, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_46 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_595 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_596 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_597 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_600 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_48 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.bfloat16)
    getitem_601 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_602 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_603 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_47 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_608 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_609 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_610 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_613 = rand_strided((25088, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_49 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_614 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_615 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_616 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_621 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_622 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_623 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_626 = rand_strided((25088, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_50 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_627 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_628 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_629 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_49 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_634 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_635 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_636 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_639 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_51 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.bfloat16)
    getitem_640 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_641 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_642 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_647 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_648 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_649 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_652 = rand_strided((25088, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_52 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_653 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_654 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_655 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_660 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_661 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_662 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_665 = rand_strided((25088, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    convert_element_type_53 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    getitem_666 = rand_strided((25088, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_667 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_668 = rand_strided((25088, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_52 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_673 = rand_strided((100352, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_674 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_675 = rand_strided((100352, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_678 = rand_strided((100352, 16), (16, 1), device='cuda:0', dtype=torch.int32)
    getitem_679 = rand_strided((2048, 32), (32, 1), device='cuda:0', dtype=torch.int32)
    getitem_680 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    getitem_681 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_54 = rand_strided((100, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((512, 100), (100, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, getitem, getitem_1, getitem_2, rsqrt, getitem_7, getitem_8, getitem_9, getitem_10, getitem_12, getitem_14, convert_element_type_2, getitem_15, getitem_16, getitem_17, rsqrt_1, getitem_22, getitem_23, getitem_24, getitem_27, convert_element_type_3, getitem_28, getitem_29, getitem_30, rsqrt_2, getitem_35, getitem_36, getitem_37, getitem_40, convert_element_type_4, getitem_41, getitem_42, getitem_43, rsqrt_3, getitem_48, getitem_49, getitem_50, convert_element_type_5, getitem_51, getitem_52, getitem_53, rsqrt_4, getitem_58, getitem_59, getitem_60, getitem_63, convert_element_type_6, getitem_64, getitem_65, getitem_66, rsqrt_5, getitem_71, getitem_72, getitem_73, getitem_76, convert_element_type_7, getitem_77, getitem_78, getitem_79, rsqrt_6, getitem_84, getitem_85, getitem_86, getitem_89, convert_element_type_8, getitem_90, getitem_91, getitem_92, rsqrt_7, getitem_97, getitem_98, getitem_99, getitem_102, convert_element_type_9, getitem_103, getitem_104, getitem_105, rsqrt_8, getitem_110, getitem_111, getitem_112, getitem_115, convert_element_type_10, getitem_116, getitem_117, getitem_118, rsqrt_9, getitem_123, getitem_124, getitem_125, getitem_128, convert_element_type_11, getitem_129, getitem_130, getitem_131, rsqrt_10, getitem_136, getitem_137, getitem_138, getitem_141, convert_element_type_12, getitem_142, getitem_143, getitem_144, rsqrt_11, getitem_149, getitem_150, getitem_151, getitem_154, convert_element_type_13, getitem_155, getitem_156, getitem_157, rsqrt_12, getitem_162, getitem_163, getitem_164, getitem_167, convert_element_type_14, getitem_168, getitem_169, getitem_170, rsqrt_13, getitem_175, getitem_176, getitem_177, convert_element_type_15, getitem_178, getitem_179, getitem_180, rsqrt_14, getitem_185, getitem_186, getitem_187, getitem_190, convert_element_type_16, getitem_191, getitem_192, getitem_193, rsqrt_15, getitem_198, getitem_199, getitem_200, getitem_203, convert_element_type_17, getitem_204, getitem_205, getitem_206, rsqrt_16, getitem_211, getitem_212, getitem_213, getitem_216, convert_element_type_18, getitem_217, getitem_218, getitem_219, rsqrt_17, getitem_224, getitem_225, getitem_226, getitem_229, convert_element_type_19, getitem_230, getitem_231, getitem_232, rsqrt_18, getitem_237, getitem_238, getitem_239, getitem_242, convert_element_type_20, getitem_243, getitem_244, getitem_245, rsqrt_19, getitem_250, getitem_251, getitem_252, getitem_255, convert_element_type_21, getitem_256, getitem_257, getitem_258, rsqrt_20, getitem_263, getitem_264, getitem_265, getitem_268, convert_element_type_22, getitem_269, getitem_270, getitem_271, rsqrt_21, getitem_276, getitem_277, getitem_278, getitem_281, convert_element_type_23, getitem_282, getitem_283, getitem_284, rsqrt_22, getitem_289, getitem_290, getitem_291, getitem_294, convert_element_type_24, getitem_295, getitem_296, getitem_297, rsqrt_23, getitem_302, getitem_303, getitem_304, getitem_307, convert_element_type_25, getitem_308, getitem_309, getitem_310, rsqrt_24, getitem_315, getitem_316, getitem_317, getitem_320, convert_element_type_26, getitem_321, getitem_322, getitem_323, rsqrt_25, getitem_328, getitem_329, getitem_330, getitem_333, convert_element_type_27, getitem_334, getitem_335, getitem_336, rsqrt_26, getitem_341, getitem_342, getitem_343, convert_element_type_28, getitem_344, getitem_345, getitem_346, rsqrt_27, getitem_351, getitem_352, getitem_353, getitem_356, convert_element_type_29, getitem_357, getitem_358, getitem_359, rsqrt_28, getitem_364, getitem_365, getitem_366, getitem_369, convert_element_type_30, getitem_370, getitem_371, getitem_372, rsqrt_29, getitem_377, getitem_378, getitem_379, getitem_382, convert_element_type_31, getitem_383, getitem_384, getitem_385, rsqrt_30, getitem_390, getitem_391, getitem_392, getitem_395, convert_element_type_32, getitem_396, getitem_397, getitem_398, rsqrt_31, getitem_403, getitem_404, getitem_405, getitem_408, convert_element_type_33, getitem_409, getitem_410, getitem_411, rsqrt_32, getitem_416, getitem_417, getitem_418, getitem_421, convert_element_type_34, getitem_422, getitem_423, getitem_424, rsqrt_33, getitem_429, getitem_430, getitem_431, getitem_434, convert_element_type_35, getitem_435, getitem_436, getitem_437, rsqrt_34, getitem_442, getitem_443, getitem_444, getitem_447, convert_element_type_36, getitem_448, getitem_449, getitem_450, rsqrt_35, getitem_455, getitem_456, getitem_457, getitem_460, convert_element_type_37, getitem_461, getitem_462, getitem_463, rsqrt_36, getitem_468, getitem_469, getitem_470, getitem_473, convert_element_type_38, getitem_474, getitem_475, getitem_476, rsqrt_37, getitem_481, getitem_482, getitem_483, getitem_486, convert_element_type_39, getitem_487, getitem_488, getitem_489, rsqrt_38, getitem_494, getitem_495, getitem_496, getitem_499, convert_element_type_40, getitem_500, getitem_501, getitem_502, rsqrt_39, getitem_507, getitem_508, getitem_509, getitem_512, convert_element_type_41, getitem_513, getitem_514, getitem_515, rsqrt_40, getitem_520, getitem_521, getitem_522, getitem_525, convert_element_type_42, getitem_526, getitem_527, getitem_528, rsqrt_41, getitem_533, getitem_534, getitem_535, getitem_538, convert_element_type_43, getitem_539, getitem_540, getitem_541, rsqrt_42, getitem_546, getitem_547, getitem_548, getitem_551, convert_element_type_44, getitem_552, getitem_553, getitem_554, rsqrt_43, getitem_559, getitem_560, getitem_561, getitem_564, convert_element_type_45, getitem_565, getitem_566, getitem_567, rsqrt_44, getitem_572, getitem_573, getitem_574, getitem_577, convert_element_type_46, getitem_578, getitem_579, getitem_580, rsqrt_45, getitem_585, getitem_586, getitem_587, convert_element_type_47, getitem_588, getitem_589, getitem_590, rsqrt_46, getitem_595, getitem_596, getitem_597, getitem_600, convert_element_type_48, getitem_601, getitem_602, getitem_603, rsqrt_47, getitem_608, getitem_609, getitem_610, getitem_613, convert_element_type_49, getitem_614, getitem_615, getitem_616, rsqrt_48, getitem_621, getitem_622, getitem_623, getitem_626, convert_element_type_50, getitem_627, getitem_628, getitem_629, rsqrt_49, getitem_634, getitem_635, getitem_636, getitem_639, convert_element_type_51, getitem_640, getitem_641, getitem_642, rsqrt_50, getitem_647, getitem_648, getitem_649, getitem_652, convert_element_type_52, getitem_653, getitem_654, getitem_655, rsqrt_51, getitem_660, getitem_661, getitem_662, getitem_665, convert_element_type_53, getitem_666, getitem_667, getitem_668, rsqrt_52, getitem_673, getitem_674, getitem_675, getitem_678, getitem_679, getitem_680, getitem_681, convert_element_type_54, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
