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


# kernel path: /tmp/torchinductor_yyu496/l4/cl46yd3g4p64274n76isokj5kq4sv5dt3nvxexsioy67uwac2g5q.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 37632, 'x': 37632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/jx/cjx5c4m6rifpslk3f5b7ky4cirawyakd6xkm4lsc6npldz76mhgc.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
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


# kernel path: /tmp/torchinductor_yyu496/5r/c5rqxmstly5qajn376vhoxkwr377bnhijndw5yjrwq2fchvetuqp.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   x_1 => convert_element_type_2, var_mean
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_2, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_2 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 826900480, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 200704
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/nl/cnleocpjhjp266bwldyyaof26ahmv6gyqcug7dhuma53v7yfitbe.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   x_1 => convert_element_type_2, var_mean
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_2, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_3 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2457600, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1600
    r0_numel = 126
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 64
    x0 = (xindex % 64)
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 126*x1
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 8064*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 64*r0_2 + 8064*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + 64*r0_2 + 8064*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_combine(
            tmp9_mean, tmp9_m2, tmp9_weight,
            tmp6, tmp7, tmp8
        )
        tmp9_mean = tl.where(r0_mask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(r0_mask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(r0_mask & xmask, tmp9_weight_next, tmp9_weight)
    tmp12, tmp13, tmp14 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp12[:, None]
    tmp10 = tmp13[:, None]
    tmp11 = tmp14[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(out_ptr2 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pm/cpmori2qpkpgukkrsai3qz7wm7pywd4emxfotxtiuz5ku33mrjuk.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   x_1 => add_1, add_2, add_3, convert_element_type_2, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, var_mean
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_2, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 0.1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, 0.9), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, 1.0000001557019536), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, 0.9), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_4, %add_2), kwargs = {})
#   %copy__2 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_5, %add_3), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__4 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 32},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__4', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 22272, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 25
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 6422528.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000001557019536
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vv/cvvqf2rzxlstfysg3oomvwhe74qa6zdeb2nxcqh5gnzzodkxmeah.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add_1, add_4, convert_element_type_2, convert_element_type_3, mul, mul_6, rsqrt, sub, var_mean
#   x_2 => relu
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_2, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_3), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2466251776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 6422528.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gb/cgbkzrpal37isvkow7qzrka4bed5tg6fmumnmaddnhe5i7expbze.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_3 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_6 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2466250752}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 3584) % 56)
    x1 = ((xindex // 64) % 56)
    x0 = (xindex % 64)
    x5 = xindex // 3584
    x6 = xindex
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + 128*x1 + 14336*x5), tmp10, other=float("-inf")).to(tl.float32)
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + 128*x1 + 14336*x5), tmp16, other=float("-inf")).to(tl.float32)
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + 128*x1 + 14336*x5), tmp23, other=float("-inf")).to(tl.float32)
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 14336*x5), tmp30, other=float("-inf")).to(tl.float32)
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 14336*x5), tmp33, other=float("-inf")).to(tl.float32)
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 14336*x5), tmp36, other=float("-inf")).to(tl.float32)
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7104 + x0 + 128*x1 + 14336*x5), tmp43, other=float("-inf")).to(tl.float32)
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 14336*x5), tmp46, other=float("-inf")).to(tl.float32)
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7232 + x0 + 128*x1 + 14336*x5), tmp49, other=float("-inf")).to(tl.float32)
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tmp52 = tmp11 > tmp17
    tmp53 = tmp11 == tmp17
    tmp54 = tmp11 != tmp11
    tmp55 = tmp17 != tmp17
    tmp56 = tmp54 > tmp55
    tmp57 = tmp52 | tmp56
    tmp58 = tmp54 & tmp55
    tmp59 = tmp53 | tmp58
    tmp60 = tl.full([1], 1, tl.int64)
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
    tmp74 = tl.full([1], 2, tl.int64)
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
    tmp88 = tl.full([1], 3, tl.int64)
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
    tmp102 = tl.full([1], 4, tl.int64)
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
    tmp116 = tl.full([1], 5, tl.int64)
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
    tmp130 = tl.full([1], 6, tl.int64)
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
    tmp144 = tl.full([1], 7, tl.int64)
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
    tmp158 = tl.full([1], 8, tl.int64)
    tmp159 = tmp149 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tmp155 | tmp160
    tmp162 = tl.where(tmp161, tmp148, tmp50)
    tmp163 = tl.where(tmp161, tmp149, tmp158)
    tmp164 = tmp163.to(tl.int8)
    tl.store(out_ptr0 + (x6), tmp51, None)
    tl.store(out_ptr1 + (x6), tmp164, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qa/cqaijod7iv2pl7ou44z6evp46wrcpo7ebnjq6hcn3ju7daaz54iw.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out => convert_element_type_4
# Graph fragment:
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/x3/cx3lczhhosrzsvgeekj5mvgu6njftfmani2rraij7k55ik7v6kxn.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_1 => convert_element_type_5, var_mean_1
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_5, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 207093760, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_8(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 1568
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 100352*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/a7/ca7jtabij26trouw5q4knkzvkrdlhgrpwappcwc7lwnedmay6ii7.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_1 => convert_element_type_5, var_mean_1
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_5, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 798720, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 8192*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 64*r0_2 + 8192*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 64*r0_2 + 8192*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/i2/ci23grxze77obenoumj47i43cwzuag3fzomagtxlxr4owazxtc2c.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_1 => add_6, add_7, add_8, convert_element_type_5, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, var_mean_1
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_5, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_3, 0.1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_10, 0.9), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 1.0000006228081046), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 0.1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_11, 0.9), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %mul_12), kwargs = {})
#   %copy__4 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_10, %add_7), kwargs = {})
#   %copy__5 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_11, %add_8), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__10 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9216, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 64*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1605632.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000006228081046
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mn/cmnb6fj532f22rhdgjnkilsm75te7ov3vfpmeflp76imstluq6zk.py
# Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   out_1 => add_6, add_9, convert_element_type_5, convert_element_type_6, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
#   out_2 => relu_1
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_5, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %getitem_5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_5), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_7), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.bfloat16), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_6,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 616563712}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1605632.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/q7/cq7kyagayjqlgw2qkur24tsy5dwgpyqwnubi5swfff6q7dnh4yey.py
# Topologically Sorted Source Nodes: [out_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_3 => convert_element_type_7
# Graph fragment:
#   %convert_element_type_7 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_14, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 147456, 'x': 147456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/3d/c3dgummirf4krqcenbaa5ntp4v2nfyrxwc2bkh6ie3g3lpoluqzs.py
# Topologically Sorted Source Nodes: [out_6], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_6 => convert_element_type_10
# Graph fragment:
#   %convert_element_type_10 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_20, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_13 = async_compile.triton('triton_poi_fused__to_copy_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/er/cerpi4lknigzs563uo3cougv7f3axzgxidvtdcy52gw6pslgxl7p.py
# Topologically Sorted Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_7 => convert_element_type_11, var_mean_3
# Graph fragment:
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_11, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 826900480, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_14(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 200704
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gf/cgfpz7qeehnmfjbh7256grbmmpm4tfq6vwq2a5mtlqdmq23fw4re.py
# Topologically Sorted Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_7 => convert_element_type_11, var_mean_3
# Graph fragment:
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_11, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2451456, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1792
    r0_numel = 112
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 28672*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 28672*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_2 + 28672*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xm/cxmcgk4bggkmipv7huyvq7qiclwna5cckahbjcuk2rnvuju2fyrx.py
# Topologically Sorted Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_7 => add_16, add_17, add_18, convert_element_type_11, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, var_mean_3
# Graph fragment:
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_11, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_9, 0.1), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_22, 0.9), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %mul_23), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_11, 1.0000006228081046), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, 0.1), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_23, 0.9), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %mul_26), kwargs = {})
#   %copy__10 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_22, %add_17), kwargs = {})
#   %copy__11 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_23, %add_18), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__16 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__16', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 33792, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 7
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_1), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1605632.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000006228081046
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/n3/cn3q7hdns3xosv3keipzifeafdu5sjlxjgjk4holo7bf7p75zbv2.py
# Topologically Sorted Source Nodes: [out_7, input_2, out_8, out_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_21, add_24, convert_element_type_14, convert_element_type_15, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
#   out_7 => add_16, add_19, convert_element_type_11, convert_element_type_12, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
#   out_8 => add_25
#   out_9 => relu_3
# Graph fragment:
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_11, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %getitem_9), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_13), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_15), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_19, torch.bfloat16), kwargs = {})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_4, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_14, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %getitem_11), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_17), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %unsqueeze_19), kwargs = {})
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_24, torch.bfloat16), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_12, %convert_element_type_15), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_25,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3288342528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1605632.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 - tmp18
    tmp21 = (tmp20 / tmp5)
    tmp22 = tmp21 + tmp7
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp19 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp15 + tmp29
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tl.store(in_out_ptr0 + (x2), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3s/c3s2d75h344dm5j4ztf6em3svmvvn5ann2itiuukr4f4gwjgxxwu.py
# Topologically Sorted Source Nodes: [out_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_17 => add_37, add_40, convert_element_type_23, convert_element_type_24, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
#   out_18 => add_41
#   out_19 => relu_6
# Graph fragment:
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_7, torch.float32), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_23, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_37,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %getitem_17), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_29), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %unsqueeze_31), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_40, torch.bfloat16), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_24, %relu_3), kwargs = {})
#   %relu_6 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3288338432}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1605632.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gy/cgyoq45rjp3v2i37vb4qvh5akatfix5m632iqoy64j454n2xajbf.py
# Topologically Sorted Source Nodes: [out_30], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_30 => convert_element_type_34
# Graph fragment:
#   %convert_element_type_34 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_68, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_19 = async_compile.triton('triton_poi_fused__to_copy_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/nf/cnfml5vk2ol7qobu7bhmqn2cnyecrttgjbju6oxg3b37uuwappkk.py
# Topologically Sorted Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_31 => convert_element_type_35, var_mean_11
# Graph fragment:
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_35, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 413450240, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_20(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 100352
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 262144*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, xmask)
    tl.store(out_ptr1 + (x3), tmp4, xmask)
    tl.store(out_ptr2 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qw/cqwqsvfg454of3qyfyohxku2r66fax6jtxm4vdhiadfuagvqc4v5.py
# Topologically Sorted Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_31 => convert_element_type_35, var_mean_11
# Graph fragment:
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_35, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1225728, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 896
    r0_numel = 112
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 14336*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 14336*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 14336*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hc/chcmwim5vrmibbzjz3oefvimd2yysanzpm76kk76au3674bjben2.py
# Topologically Sorted Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_31 => add_59, add_60, add_61, convert_element_type_35, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, var_mean_11
# Graph fragment:
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_35, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_11 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_59,), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_33, 0.1), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_70, 0.9), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_78, %mul_79), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_35, 1.0000006228081046), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_80, 0.1), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_71, 0.9), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %mul_82), kwargs = {})
#   %copy__34 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_70, %add_60), kwargs = {})
#   %copy__35 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_71, %add_61), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__22 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__22', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16896, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 7
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_1), r0_mask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1605632.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000006228081046
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ex/cexoyfybg6mt5njiquikzjdb6vcylcmvjkxtes3h6vm36h4ha4r3.py
# Topologically Sorted Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   out_31 => add_59, add_62, convert_element_type_35, convert_element_type_36, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
#   out_32 => relu_10
# Graph fragment:
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_35, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_59 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_11 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_59,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %getitem_25), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_11), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_77, %unsqueeze_45), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_47), kwargs = {})
#   %convert_element_type_36 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_62, torch.bfloat16), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_36,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1233127424}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1605632.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/tg/ctgnmndeolo7om3mdnqf46qwhbamavnkfxewxqczmbdrg4wie2er.py
# Topologically Sorted Source Nodes: [out_33], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_33 => convert_element_type_37
# Graph fragment:
#   %convert_element_type_37 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_74, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_24 = async_compile.triton('triton_poi_fused__to_copy_24', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 589824, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/ve/cvev3u5hhlznzxeiwurwxbwie4dg57xcmbit22unoz33zadtenle.py
# Topologically Sorted Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_34 => convert_element_type_38, var_mean_12
# Graph fragment:
#   %convert_element_type_38 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_12, torch.float32), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 105906176, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_25(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 392
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 50176*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bf/cbfbb56nkkdijsyjeq6tfng7gla23e4ujag3nzlbfuzwjkatwiw5.py
# Topologically Sorted Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_34 => convert_element_type_38, var_mean_12
# Graph fragment:
#   %convert_element_type_38 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_12, torch.float32), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1597440, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/d5/cd5sgl4wtf2ar5gklbkjvbvpudz5ilrcu3wsd56uppslea7celyq.py
# Topologically Sorted Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_34 => add_64, add_65, add_66, convert_element_type_38, mul_85, mul_86, mul_87, mul_88, mul_89, rsqrt_12, var_mean_12
# Graph fragment:
#   %convert_element_type_38 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_12, torch.float32), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_64,), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_36, 0.1), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_76, 0.9), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %mul_86), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_38, 1.0000024912370735), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, 0.1), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_77, 0.9), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %mul_89), kwargs = {})
#   %copy__37 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_76, %add_65), kwargs = {})
#   %copy__38 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_77, %add_66), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__27 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__27', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 401408.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000024912370735
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/72/c72kreemfsapo3blnwllgx2vrs2muplq7dzwbob425jzrbsko5yc.py
# Topologically Sorted Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   out_34 => add_64, add_67, convert_element_type_38, convert_element_type_39, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
#   out_35 => relu_11
# Graph fragment:
#   %convert_element_type_38 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_12, torch.float32), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-05), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_64,), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %getitem_27), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_12), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %unsqueeze_49), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %unsqueeze_51), kwargs = {})
#   %convert_element_type_39 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_67, torch.bfloat16), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_39,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 308283392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 401408.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sv/csv6rafuxhzzoh33ldtkwg63qohhs3gxgoplgsxslzgua4zaflix.py
# Topologically Sorted Source Nodes: [out_36], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_36 => convert_element_type_40
# Graph fragment:
#   %convert_element_type_40 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_80, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_29 = async_compile.triton('triton_poi_fused__to_copy_29', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/nu/cnuu6qxwla7n7pvsb6jlj5esh4eqroob4w3et7aduxhjsghppdz5.py
# Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_37 => convert_element_type_41, var_mean_13
# Graph fragment:
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_13, torch.float32), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_41, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 413450240, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_30(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 100352
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, xmask)
    tl.store(out_ptr1 + (x3), tmp4, xmask)
    tl.store(out_ptr2 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/su/csum7mxgnmhyx5a2v6qgvwwpyzcbam2a7bvt6sd4aphrjjpypmye.py
# Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_37 => convert_element_type_41, var_mean_13
# Graph fragment:
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_13, torch.float32), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_41, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1228800, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_31(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 98
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 50176*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 50176*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 512*r0_2 + 50176*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fz/cfzevqdadazoaun33l6r4unlblf3ibnjx42rr4rkurnopg62oafg.py
# Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_37 => add_69, add_70, add_71, convert_element_type_41, mul_92, mul_93, mul_94, mul_95, mul_96, rsqrt_13, var_mean_13
# Graph fragment:
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_13, torch.float32), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_41, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-05), kwargs = {})
#   %rsqrt_13 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_39, 0.1), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_82, 0.9), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %mul_93), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_41, 1.0000024912370735), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, 0.1), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_83, 0.9), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %mul_96), kwargs = {})
#   %copy__40 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_82, %add_70), kwargs = {})
#   %copy__41 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_83, %add_71), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__32 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__32', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 401408.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000024912370735
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mr/cmr6qq3y57r6onc32cmdekxkqqdomyg3s4r2h5vn5d6mn4ytjfka.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_3 => convert_element_type_43
# Graph fragment:
#   %convert_element_type_43 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_86, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_33 = async_compile.triton('triton_poi_fused__to_copy_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/uy/cuy2icgff3ldfzgi3oygn6wxkynotftqhdapyzlql4m2gdynsnu7.py
# Topologically Sorted Source Nodes: [out_37, input_4, out_38, out_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_74, add_77, convert_element_type_44, convert_element_type_45, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
#   out_37 => add_69, add_72, convert_element_type_41, convert_element_type_42, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
#   out_38 => add_78
#   out_39 => relu_12
# Graph fragment:
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_13, torch.float32), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_41, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-05), kwargs = {})
#   %rsqrt_13 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_69,), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %getitem_29), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_13), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_53), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %unsqueeze_55), kwargs = {})
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_72, torch.bfloat16), kwargs = {})
#   %convert_element_type_44 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_14, torch.float32), kwargs = {})
#   %var_mean_14 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_44, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-05), kwargs = {})
#   %rsqrt_14 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_74,), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %getitem_31), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %rsqrt_14), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, %unsqueeze_57), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %unsqueeze_59), kwargs = {})
#   %convert_element_type_45 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_77, torch.bfloat16), kwargs = {})
#   %add_78 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, %convert_element_type_45), kwargs = {})
#   %relu_12 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_78,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1644183552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 401408.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 - tmp18
    tmp21 = (tmp20 / tmp5)
    tmp22 = tmp21 + tmp7
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp19 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp15 + tmp29
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tl.store(in_out_ptr0 + (x2), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/62/c62fv2ym4ajfnaiabzx7ovqwajhihm7rkq2xiziwzzmffxvngcke.py
# Topologically Sorted Source Nodes: [out_47, out_48, out_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_47 => add_90, add_93, convert_element_type_53, convert_element_type_54, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
#   out_48 => add_94
#   out_49 => relu_15
# Graph fragment:
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_17, torch.float32), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_53, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_36, 1e-05), kwargs = {})
#   %rsqrt_17 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_90,), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %getitem_37), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %rsqrt_17), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %unsqueeze_69), kwargs = {})
#   %add_93 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_71), kwargs = {})
#   %convert_element_type_54 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_93, torch.bfloat16), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_54, %relu_12), kwargs = {})
#   %relu_15 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_94,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_relu_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1644175360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 401408.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vn/cvncdfnxelnubd3qfggsxmix3cwusmeeozw273zip37tvsqlzrtb.py
# Topologically Sorted Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_71 => convert_element_type_74, var_mean_24
# Graph fragment:
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_24, torch.float32), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_74, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 208666624, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_36(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 784
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 200704*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/f5/cf5tzl5vuccjpdypsgeo5cebvkunrkzs4az7dl2wm7gjkkkbucq7.py
# Topologically Sorted Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_71 => convert_element_type_74, var_mean_24
# Graph fragment:
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_24, torch.float32), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_74, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1597440, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_37(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_2 + 32768*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ki/ckiotopfdh4kvnj3nzibmmuyxsv7te4twz3yabjsogxqnd7i3uww.py
# Topologically Sorted Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_71 => add_128, add_129, add_130, convert_element_type_74, mul_169, mul_170, mul_171, mul_172, mul_173, rsqrt_24, var_mean_24
# Graph fragment:
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_24, torch.float32), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_74, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-05), kwargs = {})
#   %rsqrt_24 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_128,), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_72, 0.1), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_148, 0.9), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_169, %mul_170), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_74, 1.0000024912370735), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_171, 0.1), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_149, 0.9), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_172, %mul_173), kwargs = {})
#   %copy__73 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_148, %add_129), kwargs = {})
#   %copy__74 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_149, %add_130), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__38 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__38', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 401408.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000024912370735
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/76/c76j4laahjtq3c3l4xax3636m2swcnhok6vr6j7ieb3rq2mxkcls.py
# Topologically Sorted Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   out_71 => add_128, add_131, convert_element_type_74, convert_element_type_75, mul_168, mul_174, rsqrt_24, sub_24, var_mean_24
#   out_72 => relu_22
# Graph fragment:
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_24, torch.float32), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_74, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-05), kwargs = {})
#   %rsqrt_24 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_128,), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %getitem_51), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_168, %unsqueeze_97), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_174, %unsqueeze_99), kwargs = {})
#   %convert_element_type_75 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_131, torch.bfloat16), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_75,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 616566784}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 401408.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7f/c7f6yfzot4tumszbn3wmwoi3qepeiuwv7nyvy7mwjzaygaffhthn.py
# Topologically Sorted Source Nodes: [out_73], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_73 => convert_element_type_76
# Graph fragment:
#   %convert_element_type_76 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_152, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_40 = async_compile.triton('triton_poi_fused__to_copy_40', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_40(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/a5/ca55gktydaoxcnx77ps4bm2dcrw2tbwths244gvcj3wq6wdfljrz.py
# Topologically Sorted Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_74 => convert_element_type_77, var_mean_25
# Graph fragment:
#   %convert_element_type_77 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_25, torch.float32), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_77, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_41 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 54525952, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_41(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 196
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 50176*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/h2/ch2bv6hoxmkhfvajxrj6fqs6elilg6jo5of27zrmynrbvls5j44d.py
# Topologically Sorted Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_74 => add_133, add_134, add_135, convert_element_type_77, mul_176, mul_177, mul_178, mul_179, mul_180, rsqrt_25, var_mean_25
# Graph fragment:
#   %convert_element_type_77 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_25, torch.float32), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_77, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-05), kwargs = {})
#   %rsqrt_25 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_133,), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_75, 0.1), kwargs = {})
#   %mul_177 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_154, 0.9), kwargs = {})
#   %add_134 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %mul_177), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_77, 1.00000996502277), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, 0.1), kwargs = {})
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_155, 0.9), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %mul_180), kwargs = {})
#   %copy__76 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_154, %add_134), kwargs = {})
#   %copy__77 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_155, %add_135), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__42 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__42', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 100352.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bg/cbgpll6ppe76oljz762m5tnzomq2r3jggl32aoavlcb7rb5qehw6.py
# Topologically Sorted Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   out_74 => add_133, add_136, convert_element_type_77, convert_element_type_78, mul_175, mul_181, rsqrt_25, sub_25, var_mean_25
#   out_75 => relu_23
# Graph fragment:
#   %convert_element_type_77 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_25, torch.float32), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_77, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-05), kwargs = {})
#   %rsqrt_25 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_133,), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %getitem_53), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_25), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %unsqueeze_101), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_181, %unsqueeze_103), kwargs = {})
#   %convert_element_type_78 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_136, torch.bfloat16), kwargs = {})
#   %relu_23 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_78,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 154144768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 100352.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jr/cjr3hbetlexvo4onhijz4k3yubcvkdqn5edw5bxqucusvhl374xx.py
# Topologically Sorted Source Nodes: [out_76], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_76 => convert_element_type_79
# Graph fragment:
#   %convert_element_type_79 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_158, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_44 = async_compile.triton('triton_poi_fused__to_copy_44', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/eb/ceby3dpue7ws3y6x6txu5box4wtbg2tzltjhacchnena2tx6imfa.py
# Topologically Sorted Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_77 => convert_element_type_80, var_mean_26
# Graph fragment:
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_26, torch.float32), kwargs = {})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_80, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 208666624, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_45(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 784
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/x2/cx2xcggeliwrcknt6pwr4wwzhaovbvm77g3bxpt5jlbhwuq373c3.py
# Topologically Sorted Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_77 => add_138, add_139, add_140, convert_element_type_80, mul_183, mul_184, mul_185, mul_186, mul_187, rsqrt_26, var_mean_26
# Graph fragment:
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_26, torch.float32), kwargs = {})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_80, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_138 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_54, 1e-05), kwargs = {})
#   %rsqrt_26 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_138,), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_78, 0.1), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_160, 0.9), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_183, %mul_184), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_80, 1.00000996502277), kwargs = {})
#   %mul_186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_185, 0.1), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_161, 0.9), kwargs = {})
#   %add_140 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_186, %mul_187), kwargs = {})
#   %copy__79 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_160, %add_139), kwargs = {})
#   %copy__80 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_161, %add_140), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_copy__46 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_copy__46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_copy__46', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1622016, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_copy__46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tmp19 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = 100352.0
    tmp13 = (tmp7 / tmp12)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = 0.1
    tmp18 = tmp6 * tmp17
    tmp20 = 0.9
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = 1.00000996502277
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24 * tmp17
    tmp27 = tmp26 * tmp20
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr4 + (x0), tmp22, xmask)
    tl.store(out_ptr6 + (x0), tmp28, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/wj/cwje52vru2lr3scef4mqcphrt6vbisoarnmhixubxjj2jqbtnxwq.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_82
# Graph fragment:
#   %convert_element_type_82 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_164, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_47 = async_compile.triton('triton_poi_fused__to_copy_47', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/w4/cw4sctaqil3uitkusadf3amkausfqb62cmpikruu4dlyyfnoal5j.py
# Topologically Sorted Source Nodes: [out_77, input_6, out_78, out_79], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_143, add_146, convert_element_type_83, convert_element_type_84, mul_189, mul_195, rsqrt_27, sub_27, var_mean_27
#   out_77 => add_138, add_141, convert_element_type_80, convert_element_type_81, mul_182, mul_188, rsqrt_26, sub_26, var_mean_26
#   out_78 => add_147
#   out_79 => relu_24
# Graph fragment:
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_26, torch.float32), kwargs = {})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_80, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_138 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_54, 1e-05), kwargs = {})
#   %rsqrt_26 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_138,), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %getitem_55), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_26), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_182, %unsqueeze_105), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %unsqueeze_107), kwargs = {})
#   %convert_element_type_81 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_141, torch.bfloat16), kwargs = {})
#   %convert_element_type_83 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_27, torch.float32), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_83, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_56, 1e-05), kwargs = {})
#   %rsqrt_27 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_143,), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %getitem_57), kwargs = {})
#   %mul_189 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_27), kwargs = {})
#   %mul_195 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_189, %unsqueeze_109), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_195, %unsqueeze_111), kwargs = {})
#   %convert_element_type_84 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_146, torch.bfloat16), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_81, %convert_element_type_84), kwargs = {})
#   %relu_24 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_147,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822116352}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 100352.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 - tmp18
    tmp21 = (tmp20 / tmp5)
    tmp22 = tmp21 + tmp7
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp19 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp15 + tmp29
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tl.store(in_out_ptr0 + (x2), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/3a/c3arzdyft7aoz6cb47rhntnux2the7urgucjjhxrvm554gds4ehh.py
# Topologically Sorted Source Nodes: [out_87, out_88, out_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_87 => add_159, add_162, convert_element_type_92, convert_element_type_93, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
#   out_88 => add_163
#   out_89 => relu_27
# Graph fragment:
#   %convert_element_type_92 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_30, torch.float32), kwargs = {})
#   %var_mean_30 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_92, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_62, 1e-05), kwargs = {})
#   %rsqrt_30 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_159,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %getitem_63), kwargs = {})
#   %mul_210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %rsqrt_30), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_210, %unsqueeze_121), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_216, %unsqueeze_123), kwargs = {})
#   %convert_element_type_93 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_162, torch.bfloat16), kwargs = {})
#   %add_163 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_93, %relu_24), kwargs = {})
#   %relu_27 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_163,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822099968}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 100352.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gy/cgyy5rdxvoaahb5xr7kvt2yzqhmgsgl37t6ov6hocz6mfnmx7syv.py
# Topologically Sorted Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_131 => convert_element_type_131, var_mean_43
# Graph fragment:
#   %convert_element_type_131 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_43, torch.float32), kwargs = {})
#   %var_mean_43 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_131, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_50 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 105906176, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_50(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 392
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 200704*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/m7/cm7arwetuhjwwwkqmajzb6sojpm6oyrdcnqusoyrhc6uucurubyg.py
# Topologically Sorted Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_131 => convert_element_type_131, var_mean_43
# Graph fragment:
#   %convert_element_type_131 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_43, torch.float32), kwargs = {})
#   %var_mean_43 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_131, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1597440, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_51(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 65536*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 512*r0_2 + 65536*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp8 = tmp11[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tl.store(out_ptr2 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jm/cjmoxceogex3e447n6su7nj2vh27mx3kq7buydmly6qnnd3b6ecg.py
# Topologically Sorted Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_131 => add_229, add_230, add_231, convert_element_type_131, mul_302, mul_303, mul_304, mul_305, mul_306, rsqrt_43, var_mean_43
# Graph fragment:
#   %convert_element_type_131 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_43, torch.float32), kwargs = {})
#   %var_mean_43 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_131, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_88, 1e-05), kwargs = {})
#   %rsqrt_43 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_229,), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_129, 0.1), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_262, 0.9), kwargs = {})
#   %add_230 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_302, %mul_303), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_131, 1.00000996502277), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_304, 0.1), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_263, 0.9), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_305, %mul_306), kwargs = {})
#   %copy__130 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_262, %add_230), kwargs = {})
#   %copy__131 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_263, %add_231), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__52 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__52', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 100352.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/kc/ckcczs5637gqexo24xmjgrz3wj3e7554wd6zaul5qd4ppx45dm6o.py
# Topologically Sorted Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   out_131 => add_229, add_232, convert_element_type_131, convert_element_type_132, mul_301, mul_307, rsqrt_43, sub_43, var_mean_43
#   out_132 => relu_40
# Graph fragment:
#   %convert_element_type_131 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_43, torch.float32), kwargs = {})
#   %var_mean_43 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_131, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_88, 1e-05), kwargs = {})
#   %rsqrt_43 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_229,), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %getitem_89), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %rsqrt_43), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_301, %unsqueeze_173), kwargs = {})
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_307, %unsqueeze_175), kwargs = {})
#   %convert_element_type_132 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_232, torch.bfloat16), kwargs = {})
#   %relu_40 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_132,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_relu_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 308289536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 100352.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hf/chfsyvsjpzsonoo4b6nxa2kll6zogwmugmwl7qqtqzgtg3bbz7qj.py
# Topologically Sorted Source Nodes: [out_133], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_133 => convert_element_type_133
# Graph fragment:
#   %convert_element_type_133 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_266, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_54 = async_compile.triton('triton_poi_fused__to_copy_54', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9437184, 'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_54(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/3k/c3kmhchp43vgmzvwetmefdd2wltb2zhetsclwdsnrcemgxr4gxvd.py
# Topologically Sorted Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_134 => convert_element_type_134, var_mean_44
# Graph fragment:
#   %convert_element_type_134 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_44, torch.float32), kwargs = {})
#   %var_mean_44 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_134, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 28098560, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_55(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 100352
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, xmask)
    tl.store(out_ptr1 + (x3), tmp4, xmask)
    tl.store(out_ptr2 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7s/c7sfdb4dt4qekx6ycx3ezjgjz3df3dqx2jxjvt2dqvq4uenyoyaw.py
# Topologically Sorted Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_134 => add_234, add_235, add_236, convert_element_type_134, mul_309, mul_310, mul_311, mul_312, mul_313, rsqrt_44, var_mean_44
# Graph fragment:
#   %convert_element_type_134 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_44, torch.float32), kwargs = {})
#   %var_mean_44 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_134, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_234 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_90, 1e-05), kwargs = {})
#   %rsqrt_44 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_234,), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_132, 0.1), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_268, 0.9), kwargs = {})
#   %add_235 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_309, %mul_310), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_134, 1.0000398612827361), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_311, 0.1), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_269, 0.9), kwargs = {})
#   %add_236 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_312, %mul_313), kwargs = {})
#   %copy__133 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_268, %add_235), kwargs = {})
#   %copy__134 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_269, %add_236), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__56 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__56', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 25088.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gt/cgtwdc3ssa6ghbxn7aq2yhwsjeld52lpvlevzsvfle336m4dyscu.py
# Topologically Sorted Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# Source node to ATen node mapping:
#   out_134 => add_234, add_237, convert_element_type_134, convert_element_type_135, mul_308, mul_314, rsqrt_44, sub_44, var_mean_44
#   out_135 => relu_41
# Graph fragment:
#   %convert_element_type_134 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_44, torch.float32), kwargs = {})
#   %var_mean_44 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_134, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_234 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_90, 1e-05), kwargs = {})
#   %rsqrt_44 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_234,), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %getitem_91), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %rsqrt_44), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_308, %unsqueeze_177), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_314, %unsqueeze_179), kwargs = {})
#   %convert_element_type_135 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_237, torch.bfloat16), kwargs = {})
#   %relu_41 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_135,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_relu_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 77078528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 25088.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ah/cah4mgtbsvcgxalukp4fnxtqcgickub3wv67vfve2dsx6qblwm2c.py
# Topologically Sorted Source Nodes: [out_136], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out_136 => convert_element_type_136
# Graph fragment:
#   %convert_element_type_136 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_272, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_58 = async_compile.triton('triton_poi_fused__to_copy_58', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_58(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/au/cauuwnvgxfw6nzdugyfxaoijhy6jvgflx3yl4s4advqlld6cyoqp.py
# Topologically Sorted Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_137 => convert_element_type_137, var_mean_45
# Graph fragment:
#   %convert_element_type_137 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_45, torch.float32), kwargs = {})
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_137, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__native_batch_norm_legit_functional_59 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 105906176, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_59(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 392
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hj/chjjdsq4k76ypddu4ficvk33r7i75znvzoshz5hh7b5rog5utmht.py
# Topologically Sorted Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   out_137 => add_239, add_240, add_241, convert_element_type_137, mul_316, mul_317, mul_318, mul_319, mul_320, rsqrt_45, var_mean_45
# Graph fragment:
#   %convert_element_type_137 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_45, torch.float32), kwargs = {})
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_137, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_92, 1e-05), kwargs = {})
#   %rsqrt_45 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_239,), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_135, 0.1), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_274, 0.9), kwargs = {})
#   %add_240 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_316, %mul_317), kwargs = {})
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_137, 1.0000398612827361), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_318, 0.1), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_275, 0.9), kwargs = {})
#   %add_241 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_319, %mul_320), kwargs = {})
#   %copy__136 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_274, %add_240), kwargs = {})
#   %copy__137 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_275, %add_241), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_copy__60 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__60', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1671168, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 2048*r0_1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 2048*r0_1), xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 25088.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7m/c7m6m7l4ko62eb5gra6d5wnrzlvjev5lc37sskgvcsbpdfkcyhyo.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_7 => convert_element_type_139
# Graph fragment:
#   %convert_element_type_139 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_278, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_61 = async_compile.triton('triton_poi_fused__to_copy_61', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_61(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mv/cmvqd2vfqu7rrmqfbtsngydojggwailqpiw32z5u6b4iwdkojpti.py
# Topologically Sorted Source Nodes: [out_137, input_8, out_138, out_139], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_8 => add_244, add_247, convert_element_type_140, convert_element_type_141, mul_322, mul_328, rsqrt_46, sub_46, var_mean_46
#   out_137 => add_239, add_242, convert_element_type_137, convert_element_type_138, mul_315, mul_321, rsqrt_45, sub_45, var_mean_45
#   out_138 => add_248
#   out_139 => relu_42
# Graph fragment:
#   %convert_element_type_137 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_45, torch.float32), kwargs = {})
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_137, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_239 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_92, 1e-05), kwargs = {})
#   %rsqrt_45 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_239,), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %getitem_93), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %rsqrt_45), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_315, %unsqueeze_181), kwargs = {})
#   %add_242 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_321, %unsqueeze_183), kwargs = {})
#   %convert_element_type_138 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_242, torch.bfloat16), kwargs = {})
#   %convert_element_type_140 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_46, torch.float32), kwargs = {})
#   %var_mean_46 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_140, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_244 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_94, 1e-05), kwargs = {})
#   %rsqrt_46 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_244,), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %getitem_95), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %rsqrt_46), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_322, %unsqueeze_185), kwargs = {})
#   %add_247 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_328, %unsqueeze_187), kwargs = {})
#   %convert_element_type_141 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_247, torch.bfloat16), kwargs = {})
#   %add_248 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_138, %convert_element_type_141), kwargs = {})
#   %relu_42 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_248,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_relu_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411107328}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 25088.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 - tmp18
    tmp21 = (tmp20 / tmp5)
    tmp22 = tmp21 + tmp7
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp19 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp15 + tmp29
    tmp31 = tl.full([1], 0, tl.int32)
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tl.store(in_out_ptr0 + (x2), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/nm/cnmeowf7fn6ffdlmbgfhlj5g2nebehdxefey6xp7coon4iz6zq2j.py
# Topologically Sorted Source Nodes: [out_147, out_148, out_149], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_147 => add_260, add_263, convert_element_type_149, convert_element_type_150, mul_343, mul_349, rsqrt_49, sub_49, var_mean_49
#   out_148 => add_264
#   out_149 => relu_45
# Graph fragment:
#   %convert_element_type_149 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_49, torch.float32), kwargs = {})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_149, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_260 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_100, 1e-05), kwargs = {})
#   %rsqrt_49 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_260,), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %getitem_101), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %rsqrt_49), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_197), kwargs = {})
#   %add_263 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_349, %unsqueeze_199), kwargs = {})
#   %convert_element_type_150 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_263, torch.bfloat16), kwargs = {})
#   %add_264 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_150, %relu_42), kwargs = {})
#   %relu_45 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_264,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_relu_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411074560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_63(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 25088.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lv/clvvy2laszr3dscr6xbhahpgnxxpdwmxlm6rxbhk32mguy6q2wjq.py
# Topologically Sorted Source Nodes: [out_157, out_158, out_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_157 => add_276, add_279, convert_element_type_158, convert_element_type_159, mul_364, mul_370, rsqrt_52, sub_52, var_mean_52
#   out_158 => add_280
#   out_159 => relu_48
# Graph fragment:
#   %convert_element_type_158 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_52, torch.float32), kwargs = {})
#   %var_mean_52 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_158, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_276 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_106, 1e-05), kwargs = {})
#   %rsqrt_52 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_276,), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %getitem_107), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %rsqrt_52), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_364, %unsqueeze_209), kwargs = {})
#   %add_279 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_370, %unsqueeze_211), kwargs = {})
#   %convert_element_type_159 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_279, torch.bfloat16), kwargs = {})
#   %add_280 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_159, %relu_45), kwargs = {})
#   %relu_48 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_280,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_48, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_64 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 513835008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 25088.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = 0.0
    tmp21 = tmp19 <= tmp20
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/34/c34oiklgofsrd7tnlxo23dbhnt663awwaw6j5mirgn7aehneylll.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_4 => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_48, [-1, -2], True), kwargs = {})
triton_per_fused_mean_65 = async_compile.triton('triton_per_fused_mean_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_65', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 106954752, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_mean_65(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_2 + 100352*x1), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = 49.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fn/cfn3rdhvzhllsxguva6rogwjskyp6wkjxcxfw3ro2twmzw6algc6.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_6 => convert_element_type_161
# Graph fragment:
#   %convert_element_type_161 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_320, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_66 = async_compile.triton('triton_poi_fused__to_copy_66', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_66', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1638400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_66(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2s/c2sfw3qe6d7nlko2jr4bne4cb7w37i32yvic2uysycvnacvkpq4d.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_6 => convert_element_type_160
# Graph fragment:
#   %convert_element_type_160 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_321, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_67 = async_compile.triton('triton_poi_fused__to_copy_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_67', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_67(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ey/ceyexzq2ydiikr5a3dobhdqhr7n5mfnfncu2qhajrr7hjfexpgc3.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, 1), kwargs = {})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_3, %add), kwargs = {})
triton_poi_fused_add_copy__68 = async_compile.triton('triton_poi_fused_add_copy__68', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__68', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__68(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321 = args
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
    assert_size_stride(primals_321, (100, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_1, buf0, 192, 49, stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((512, 3, 224, 224), (150528, 1, 672, 3), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_2, buf1, 1536, 50176, stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (512, 64, 112, 112), (802816, 1, 7168, 64), 'torch.ops.aten.convolution.default')
        buf3 = empty_strided_cuda((1, 64, 1, 1, 3136), (200704, 1, 200704, 200704, 64), torch.float32)
        buf4 = empty_strided_cuda((1, 64, 1, 1, 3136), (200704, 1, 200704, 200704, 64), torch.float32)
        buf5 = empty_strided_cuda((1, 64, 1, 1, 3136), (200704, 1, 200704, 200704, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_2.run(buf2, buf3, buf4, buf5, 200704, 2048, stream=stream0)
        buf6 = empty_strided_cuda((1, 64, 1, 1, 25), (1600, 1, 1600, 1600, 64), torch.float32)
        buf7 = empty_strided_cuda((1, 64, 1, 1, 25), (1600, 1, 1600, 1600, 64), torch.float32)
        buf8 = empty_strided_cuda((1, 64, 1, 1, 25), (1600, 1, 1600, 1600, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf3, buf4, buf5, buf6, buf7, buf8, 1600, 126, stream=stream0)
        buf9 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf10 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf12 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__4.run(buf6, buf7, buf8, primals_4, primals_5, buf9, buf10, buf12, primals_4, primals_5, 64, 25, stream=stream0)
        del buf6
        del buf7
        del buf8
        del primals_4
        del primals_5
        buf13 = empty_strided_cuda((512, 64, 112, 112), (802816, 1, 7168, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf2, buf9, buf10, primals_6, primals_7, buf13, 411041792, stream=stream0)
        del primals_7
        buf14 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        buf15 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.int8)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_6.run(buf13, buf14, buf15, 102760448, stream=stream0)
        buf16 = empty_strided_cuda((64, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(primals_8, buf16, 4096, stream=stream0)
        del primals_8
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf14, buf16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf18 = empty_strided_cuda((1, 64, 1, 1, 1024), (65536, 1, 65536, 65536, 64), torch.float32)
        buf19 = empty_strided_cuda((1, 64, 1, 1, 1024), (65536, 1, 65536, 65536, 64), torch.float32)
        buf20 = empty_strided_cuda((1, 64, 1, 1, 1024), (65536, 1, 65536, 65536, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf17, buf18, buf19, buf20, 65536, 1568, stream=stream0)
        buf21 = empty_strided_cuda((1, 64, 1, 1, 8), (512, 1, 512, 512, 64), torch.float32)
        buf22 = empty_strided_cuda((1, 64, 1, 1, 8), (512, 1, 512, 512, 64), torch.float32)
        buf23 = empty_strided_cuda((1, 64, 1, 1, 8), (512, 1, 512, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf18, buf19, buf20, buf21, buf22, buf23, 512, 128, stream=stream0)
        buf24 = buf10; del buf10  # reuse
        buf25 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf27 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__10.run(buf21, buf22, buf23, primals_10, primals_11, buf24, buf25, buf27, primals_10, primals_11, 64, 8, stream=stream0)
        del primals_10
        del primals_11
        buf28 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_11.run(buf17, buf24, buf25, primals_12, primals_13, buf28, 102760448, stream=stream0)
        del primals_13
        buf29 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(primals_14, buf29, 4096, 9, stream=stream0)
        del primals_14
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf31 = buf20; del buf20  # reuse
        buf32 = buf19; del buf19  # reuse
        buf33 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf30, buf31, buf32, buf33, 65536, 1568, stream=stream0)
        buf34 = buf23; del buf23  # reuse
        buf35 = buf22; del buf22  # reuse
        buf36 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf31, buf32, buf33, buf34, buf35, buf36, 512, 128, stream=stream0)
        buf37 = buf25; del buf25  # reuse
        buf38 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf40 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__10.run(buf34, buf35, buf36, primals_16, primals_17, buf37, buf38, buf40, primals_16, primals_17, 64, 8, stream=stream0)
        del primals_16
        del primals_17
        buf41 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_11.run(buf30, buf37, buf38, primals_18, primals_19, buf41, 102760448, stream=stream0)
        del primals_19
        buf42 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(primals_20, buf42, 16384, stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf41, buf42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf44 = reinterpret_tensor(buf5, (1, 256, 1, 1, 784), (200704, 1, 200704, 200704, 256), 0); del buf5  # reuse
        buf45 = reinterpret_tensor(buf4, (1, 256, 1, 1, 784), (200704, 1, 200704, 200704, 256), 0); del buf4  # reuse
        buf46 = reinterpret_tensor(buf3, (1, 256, 1, 1, 784), (200704, 1, 200704, 200704, 256), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf43, buf44, buf45, buf46, 200704, 2048, stream=stream0)
        buf47 = empty_strided_cuda((1, 256, 1, 1, 7), (1792, 1, 1792, 1792, 256), torch.float32)
        buf48 = empty_strided_cuda((1, 256, 1, 1, 7), (1792, 1, 1792, 1792, 256), torch.float32)
        buf49 = empty_strided_cuda((1, 256, 1, 1, 7), (1792, 1, 1792, 1792, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf44, buf45, buf46, buf47, buf48, buf49, 1792, 112, stream=stream0)
        buf50 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf51 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf53 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__16.run(buf47, buf48, buf49, primals_22, primals_23, buf50, buf51, buf53, primals_22, primals_23, 256, 7, stream=stream0)
        del primals_22
        del primals_23
        buf54 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(primals_26, buf54, 16384, stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf14, buf54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf56 = buf46; del buf46  # reuse
        buf57 = buf45; del buf45  # reuse
        buf58 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf55, buf56, buf57, buf58, 200704, 2048, stream=stream0)
        buf59 = buf49; del buf49  # reuse
        buf60 = buf48; del buf48  # reuse
        buf61 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf56, buf57, buf58, buf59, buf60, buf61, 1792, 112, stream=stream0)
        buf62 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf63 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf65 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__16.run(buf59, buf60, buf61, primals_28, primals_29, buf62, buf63, buf65, primals_28, primals_29, 256, 7, stream=stream0)
        del primals_28
        del primals_29
        buf66 = empty_strided_cuda((512, 256, 56, 56), (802816, 1, 14336, 256), torch.bfloat16)
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [out_7, input_2, out_8, out_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_17.run(buf67, buf43, buf50, buf51, primals_24, primals_25, buf55, buf62, buf63, primals_30, primals_31, 411041792, stream=stream0)
        del primals_25
        del primals_31
        buf68 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(primals_32, buf68, 16384, stream=stream0)
        del primals_32
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf67, buf68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf70 = buf33; del buf33  # reuse
        buf71 = buf32; del buf32  # reuse
        buf72 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf69, buf70, buf71, buf72, 65536, 1568, stream=stream0)
        buf73 = buf36; del buf36  # reuse
        buf74 = buf35; del buf35  # reuse
        buf75 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf70, buf71, buf72, buf73, buf74, buf75, 512, 128, stream=stream0)
        buf76 = buf38; del buf38  # reuse
        buf77 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf79 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__10.run(buf73, buf74, buf75, primals_34, primals_35, buf76, buf77, buf79, primals_34, primals_35, 64, 8, stream=stream0)
        del primals_34
        del primals_35
        buf80 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_11.run(buf69, buf76, buf77, primals_36, primals_37, buf80, 102760448, stream=stream0)
        del primals_37
        buf81 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(primals_38, buf81, 4096, 9, stream=stream0)
        del primals_38
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf80, buf81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf83 = buf72; del buf72  # reuse
        buf84 = buf71; del buf71  # reuse
        buf85 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf82, buf83, buf84, buf85, 65536, 1568, stream=stream0)
        buf86 = buf75; del buf75  # reuse
        buf87 = buf74; del buf74  # reuse
        buf88 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf83, buf84, buf85, buf86, buf87, buf88, 512, 128, stream=stream0)
        buf89 = buf77; del buf77  # reuse
        buf90 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf92 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__10.run(buf86, buf87, buf88, primals_40, primals_41, buf89, buf90, buf92, primals_40, primals_41, 64, 8, stream=stream0)
        del primals_40
        del primals_41
        buf93 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_11.run(buf82, buf89, buf90, primals_42, primals_43, buf93, 102760448, stream=stream0)
        del primals_43
        buf94 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(primals_44, buf94, 16384, stream=stream0)
        del primals_44
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf93, buf94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf96 = buf58; del buf58  # reuse
        buf97 = buf57; del buf57  # reuse
        buf98 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf95, buf96, buf97, buf98, 200704, 2048, stream=stream0)
        buf99 = buf61; del buf61  # reuse
        buf100 = buf60; del buf60  # reuse
        buf101 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf96, buf97, buf98, buf99, buf100, buf101, 1792, 112, stream=stream0)
        buf102 = buf63; del buf63  # reuse
        buf103 = buf51; del buf51  # reuse
        buf105 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__16.run(buf99, buf100, buf101, primals_46, primals_47, buf102, buf103, buf105, primals_46, primals_47, 256, 7, stream=stream0)
        del primals_46
        del primals_47
        buf106 = empty_strided_cuda((512, 256, 56, 56), (802816, 1, 14336, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_18.run(buf95, buf102, buf103, primals_48, primals_49, buf67, buf106, 411041792, stream=stream0)
        del primals_49
        buf107 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(primals_50, buf107, 16384, stream=stream0)
        del primals_50
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf106, buf107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf109 = buf85; del buf85  # reuse
        buf110 = buf84; del buf84  # reuse
        buf111 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf108, buf109, buf110, buf111, 65536, 1568, stream=stream0)
        buf112 = buf88; del buf88  # reuse
        buf113 = buf87; del buf87  # reuse
        buf114 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf109, buf110, buf111, buf112, buf113, buf114, 512, 128, stream=stream0)
        buf115 = buf90; del buf90  # reuse
        buf116 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf118 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__10.run(buf112, buf113, buf114, primals_52, primals_53, buf115, buf116, buf118, primals_52, primals_53, 64, 8, stream=stream0)
        del primals_52
        del primals_53
        buf119 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_11.run(buf108, buf115, buf116, primals_54, primals_55, buf119, 102760448, stream=stream0)
        del primals_55
        buf120 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(primals_56, buf120, 4096, 9, stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [out_23], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf119, buf120, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        buf122 = buf111; del buf111  # reuse
        buf123 = buf110; del buf110  # reuse
        buf124 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf121, buf122, buf123, buf124, 65536, 1568, stream=stream0)
        buf125 = buf114; del buf114  # reuse
        buf126 = buf113; del buf113  # reuse
        buf127 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf122, buf123, buf124, buf125, buf126, buf127, 512, 128, stream=stream0)
        del buf122
        del buf123
        del buf124
        buf128 = buf116; del buf116  # reuse
        buf129 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf131 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__10.run(buf125, buf126, buf127, primals_58, primals_59, buf128, buf129, buf131, primals_58, primals_59, 64, 8, stream=stream0)
        del primals_58
        del primals_59
        buf132 = empty_strided_cuda((512, 64, 56, 56), (200704, 1, 3584, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_11.run(buf121, buf128, buf129, primals_60, primals_61, buf132, 102760448, stream=stream0)
        del buf129
        del primals_61
        buf133 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_13.run(primals_62, buf133, 16384, stream=stream0)
        del primals_62
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf132, buf133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution.default')
        buf135 = buf98; del buf98  # reuse
        buf136 = buf97; del buf97  # reuse
        buf137 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf134, buf135, buf136, buf137, 200704, 2048, stream=stream0)
        buf138 = buf99; del buf99  # reuse
        buf139 = buf101; del buf101  # reuse
        buf140 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf135, buf136, buf137, buf138, buf139, buf140, 1792, 112, stream=stream0)
        del buf135
        del buf136
        del buf137
        buf141 = buf103; del buf103  # reuse
        buf142 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf144 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_27], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__16.run(buf138, buf139, buf140, primals_64, primals_65, buf141, buf142, buf144, primals_64, primals_65, 256, 7, stream=stream0)
        del buf138
        del buf139
        del buf140
        del primals_64
        del primals_65
        buf145 = empty_strided_cuda((512, 256, 56, 56), (802816, 1, 14336, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_27, out_28, out_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_18.run(buf134, buf141, buf142, primals_66, primals_67, buf106, buf145, 411041792, stream=stream0)
        del primals_67
        buf146 = empty_strided_cuda((128, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_19.run(primals_68, buf146, 32768, stream=stream0)
        del primals_68
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf145, buf146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (512, 128, 56, 56), (401408, 1, 7168, 128), 'torch.ops.aten.convolution.default')
        buf148 = empty_strided_cuda((1, 128, 1, 1, 784), (100352, 1, 100352, 100352, 128), torch.float32)
        buf149 = empty_strided_cuda((1, 128, 1, 1, 784), (100352, 1, 100352, 100352, 128), torch.float32)
        buf150 = empty_strided_cuda((1, 128, 1, 1, 784), (100352, 1, 100352, 100352, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf147, buf148, buf149, buf150, 100352, 2048, stream=stream0)
        buf151 = empty_strided_cuda((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), torch.float32)
        buf152 = empty_strided_cuda((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), torch.float32)
        buf153 = empty_strided_cuda((1, 128, 1, 1, 7), (896, 1, 896, 896, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf148, buf149, buf150, buf151, buf152, buf153, 896, 112, stream=stream0)
        buf154 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf155 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf157 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__22.run(buf151, buf152, buf153, primals_70, primals_71, buf154, buf155, buf157, primals_70, primals_71, 128, 7, stream=stream0)
        del buf151
        del buf152
        del buf153
        del primals_70
        del primals_71
        buf158 = empty_strided_cuda((512, 128, 56, 56), (401408, 1, 7168, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_23.run(buf147, buf154, buf155, primals_72, primals_73, buf158, 205520896, stream=stream0)
        del primals_73
        buf159 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(primals_74, buf159, 16384, 9, stream=stream0)
        del primals_74
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf158, buf159, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf161 = empty_strided_cuda((1, 128, 1, 1, 1024), (131072, 1, 131072, 131072, 128), torch.float32)
        buf162 = empty_strided_cuda((1, 128, 1, 1, 1024), (131072, 1, 131072, 131072, 128), torch.float32)
        buf163 = empty_strided_cuda((1, 128, 1, 1, 1024), (131072, 1, 131072, 131072, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf160, buf161, buf162, buf163, 131072, 392, stream=stream0)
        buf164 = empty_strided_cuda((1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), torch.float32)
        buf165 = empty_strided_cuda((1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), torch.float32)
        buf166 = empty_strided_cuda((1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf161, buf162, buf163, buf164, buf165, buf166, 1024, 128, stream=stream0)
        buf167 = buf155; del buf155  # reuse
        buf168 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf170 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__27.run(buf164, buf165, buf166, primals_76, primals_77, buf167, buf168, buf170, primals_76, primals_77, 128, 8, stream=stream0)
        del primals_76
        del primals_77
        buf171 = empty_strided_cuda((512, 128, 28, 28), (100352, 1, 3584, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf160, buf167, buf168, primals_78, primals_79, buf171, 51380224, stream=stream0)
        del primals_79
        buf172 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_36], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(primals_80, buf172, 65536, stream=stream0)
        del primals_80
        # Topologically Sorted Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf171, buf172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf174 = reinterpret_tensor(buf150, (1, 512, 1, 1, 196), (100352, 1, 100352, 100352, 512), 0); del buf150  # reuse
        buf175 = reinterpret_tensor(buf149, (1, 512, 1, 1, 196), (100352, 1, 100352, 100352, 512), 0); del buf149  # reuse
        buf176 = reinterpret_tensor(buf148, (1, 512, 1, 1, 196), (100352, 1, 100352, 100352, 512), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf173, buf174, buf175, buf176, 100352, 2048, stream=stream0)
        buf177 = reinterpret_tensor(buf166, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf166  # reuse
        buf178 = reinterpret_tensor(buf165, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf165  # reuse
        buf179 = reinterpret_tensor(buf164, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf174, buf175, buf176, buf177, buf178, buf179, 1024, 98, stream=stream0)
        buf180 = reinterpret_tensor(buf127, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf127  # reuse
        buf181 = reinterpret_tensor(buf126, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf126  # reuse
        buf183 = reinterpret_tensor(buf125, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__32.run(buf177, buf178, buf179, primals_82, primals_83, buf180, buf181, buf183, primals_82, primals_83, 512, 2, stream=stream0)
        del primals_82
        del primals_83
        buf184 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(primals_86, buf184, 131072, stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf145, buf184, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf186 = buf176; del buf176  # reuse
        buf187 = buf175; del buf175  # reuse
        buf188 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf185, buf186, buf187, buf188, 100352, 2048, stream=stream0)
        buf189 = buf179; del buf179  # reuse
        buf190 = buf178; del buf178  # reuse
        buf191 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf186, buf187, buf188, buf189, buf190, buf191, 1024, 98, stream=stream0)
        buf192 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf193 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf195 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__32.run(buf189, buf190, buf191, primals_88, primals_89, buf192, buf193, buf195, primals_88, primals_89, 512, 2, stream=stream0)
        del primals_88
        del primals_89
        buf196 = empty_strided_cuda((512, 512, 28, 28), (401408, 1, 14336, 512), torch.bfloat16)
        buf197 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [out_37, input_4, out_38, out_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_34.run(buf197, buf173, buf180, buf181, primals_84, primals_85, buf185, buf192, buf193, primals_90, primals_91, 205520896, stream=stream0)
        del primals_85
        del primals_91
        buf198 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_40], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(primals_92, buf198, 65536, stream=stream0)
        del primals_92
        # Topologically Sorted Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf197, buf198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf200 = buf163; del buf163  # reuse
        buf201 = buf162; del buf162  # reuse
        buf202 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf199, buf200, buf201, buf202, 131072, 392, stream=stream0)
        buf203 = reinterpret_tensor(buf191, (1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), 0); del buf191  # reuse
        buf204 = reinterpret_tensor(buf190, (1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), 0); del buf190  # reuse
        buf205 = reinterpret_tensor(buf189, (1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf200, buf201, buf202, buf203, buf204, buf205, 1024, 128, stream=stream0)
        buf206 = buf168; del buf168  # reuse
        buf207 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf209 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__27.run(buf203, buf204, buf205, primals_94, primals_95, buf206, buf207, buf209, primals_94, primals_95, 128, 8, stream=stream0)
        del primals_94
        del primals_95
        buf210 = empty_strided_cuda((512, 128, 28, 28), (100352, 1, 3584, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf199, buf206, buf207, primals_96, primals_97, buf210, 51380224, stream=stream0)
        del primals_97
        buf211 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_43], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(primals_98, buf211, 16384, 9, stream=stream0)
        del primals_98
        # Topologically Sorted Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf210, buf211, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf213 = buf202; del buf202  # reuse
        buf214 = buf201; del buf201  # reuse
        buf215 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf212, buf213, buf214, buf215, 131072, 392, stream=stream0)
        buf216 = buf205; del buf205  # reuse
        buf217 = buf204; del buf204  # reuse
        buf218 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf213, buf214, buf215, buf216, buf217, buf218, 1024, 128, stream=stream0)
        buf219 = buf207; del buf207  # reuse
        buf220 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf222 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__27.run(buf216, buf217, buf218, primals_100, primals_101, buf219, buf220, buf222, primals_100, primals_101, 128, 8, stream=stream0)
        del primals_100
        del primals_101
        buf223 = empty_strided_cuda((512, 128, 28, 28), (100352, 1, 3584, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_44, out_45], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf212, buf219, buf220, primals_102, primals_103, buf223, 51380224, stream=stream0)
        del primals_103
        buf224 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(primals_104, buf224, 65536, stream=stream0)
        del primals_104
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf223, buf224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf226 = buf188; del buf188  # reuse
        buf227 = buf187; del buf187  # reuse
        buf228 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf225, buf226, buf227, buf228, 100352, 2048, stream=stream0)
        buf229 = reinterpret_tensor(buf218, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf218  # reuse
        buf230 = reinterpret_tensor(buf217, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf217  # reuse
        buf231 = reinterpret_tensor(buf216, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf226, buf227, buf228, buf229, buf230, buf231, 1024, 98, stream=stream0)
        buf232 = buf193; del buf193  # reuse
        buf233 = buf181; del buf181  # reuse
        buf235 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_47], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__32.run(buf229, buf230, buf231, primals_106, primals_107, buf232, buf233, buf235, primals_106, primals_107, 512, 2, stream=stream0)
        del primals_106
        del primals_107
        buf236 = empty_strided_cuda((512, 512, 28, 28), (401408, 1, 14336, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_47, out_48, out_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_35.run(buf225, buf232, buf233, primals_108, primals_109, buf197, buf236, 205520896, stream=stream0)
        del primals_109
        buf237 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(primals_110, buf237, 65536, stream=stream0)
        del primals_110
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf236, buf237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf239 = buf215; del buf215  # reuse
        buf240 = buf214; del buf214  # reuse
        buf241 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf238, buf239, buf240, buf241, 131072, 392, stream=stream0)
        buf242 = reinterpret_tensor(buf231, (1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), 0); del buf231  # reuse
        buf243 = reinterpret_tensor(buf230, (1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), 0); del buf230  # reuse
        buf244 = reinterpret_tensor(buf229, (1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf239, buf240, buf241, buf242, buf243, buf244, 1024, 128, stream=stream0)
        buf245 = buf220; del buf220  # reuse
        buf246 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf248 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_51], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__27.run(buf242, buf243, buf244, primals_112, primals_113, buf245, buf246, buf248, primals_112, primals_113, 128, 8, stream=stream0)
        del primals_112
        del primals_113
        buf249 = empty_strided_cuda((512, 128, 28, 28), (100352, 1, 3584, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_51, out_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf238, buf245, buf246, primals_114, primals_115, buf249, 51380224, stream=stream0)
        del primals_115
        buf250 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_53], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(primals_116, buf250, 16384, 9, stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [out_53], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf249, buf250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf252 = buf241; del buf241  # reuse
        buf253 = buf240; del buf240  # reuse
        buf254 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf251, buf252, buf253, buf254, 131072, 392, stream=stream0)
        buf255 = buf244; del buf244  # reuse
        buf256 = buf243; del buf243  # reuse
        buf257 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf252, buf253, buf254, buf255, buf256, buf257, 1024, 128, stream=stream0)
        buf258 = buf246; del buf246  # reuse
        buf259 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf261 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__27.run(buf255, buf256, buf257, primals_118, primals_119, buf258, buf259, buf261, primals_118, primals_119, 128, 8, stream=stream0)
        del primals_118
        del primals_119
        buf262 = empty_strided_cuda((512, 128, 28, 28), (100352, 1, 3584, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_54, out_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf251, buf258, buf259, primals_120, primals_121, buf262, 51380224, stream=stream0)
        del primals_121
        buf263 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_56], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(primals_122, buf263, 65536, stream=stream0)
        del primals_122
        # Topologically Sorted Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf262, buf263, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf265 = buf228; del buf228  # reuse
        buf266 = buf227; del buf227  # reuse
        buf267 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf264, buf265, buf266, buf267, 100352, 2048, stream=stream0)
        buf268 = reinterpret_tensor(buf257, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf257  # reuse
        buf269 = reinterpret_tensor(buf256, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf256  # reuse
        buf270 = reinterpret_tensor(buf255, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf265, buf266, buf267, buf268, buf269, buf270, 1024, 98, stream=stream0)
        buf271 = buf233; del buf233  # reuse
        buf272 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf274 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__32.run(buf268, buf269, buf270, primals_124, primals_125, buf271, buf272, buf274, primals_124, primals_125, 512, 2, stream=stream0)
        del primals_124
        del primals_125
        buf275 = empty_strided_cuda((512, 512, 28, 28), (401408, 1, 14336, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_57, out_58, out_59], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_35.run(buf264, buf271, buf272, primals_126, primals_127, buf236, buf275, 205520896, stream=stream0)
        del primals_127
        buf276 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_60], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(primals_128, buf276, 65536, stream=stream0)
        del primals_128
        # Topologically Sorted Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf275, buf276, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf278 = buf254; del buf254  # reuse
        buf279 = buf253; del buf253  # reuse
        buf280 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf277, buf278, buf279, buf280, 131072, 392, stream=stream0)
        buf281 = reinterpret_tensor(buf270, (1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), 0); del buf270  # reuse
        buf282 = reinterpret_tensor(buf269, (1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), 0); del buf269  # reuse
        buf283 = reinterpret_tensor(buf268, (1, 128, 1, 1, 8), (1024, 1, 1024, 1024, 128), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf278, buf279, buf280, buf281, buf282, buf283, 1024, 128, stream=stream0)
        buf284 = buf259; del buf259  # reuse
        buf285 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf287 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__27.run(buf281, buf282, buf283, primals_130, primals_131, buf284, buf285, buf287, primals_130, primals_131, 128, 8, stream=stream0)
        del primals_130
        del primals_131
        buf288 = empty_strided_cuda((512, 128, 28, 28), (100352, 1, 3584, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_61, out_62], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf277, buf284, buf285, primals_132, primals_133, buf288, 51380224, stream=stream0)
        del primals_133
        buf289 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_63], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(primals_134, buf289, 16384, 9, stream=stream0)
        del primals_134
        # Topologically Sorted Source Nodes: [out_63], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf288, buf289, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        buf291 = buf280; del buf280  # reuse
        buf292 = buf279; del buf279  # reuse
        buf293 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf290, buf291, buf292, buf293, 131072, 392, stream=stream0)
        buf294 = buf283; del buf283  # reuse
        buf295 = buf282; del buf282  # reuse
        buf296 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf291, buf292, buf293, buf294, buf295, buf296, 1024, 128, stream=stream0)
        buf297 = buf285; del buf285  # reuse
        buf298 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf300 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__27.run(buf294, buf295, buf296, primals_136, primals_137, buf297, buf298, buf300, primals_136, primals_137, 128, 8, stream=stream0)
        del primals_136
        del primals_137
        buf301 = empty_strided_cuda((512, 128, 28, 28), (100352, 1, 3584, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_64, out_65], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_28.run(buf290, buf297, buf298, primals_138, primals_139, buf301, 51380224, stream=stream0)
        del buf298
        del primals_139
        buf302 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_66], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_29.run(primals_140, buf302, 65536, stream=stream0)
        del primals_140
        # Topologically Sorted Source Nodes: [out_66], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf301, buf302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        buf304 = buf267; del buf267  # reuse
        buf305 = buf266; del buf266  # reuse
        buf306 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf303, buf304, buf305, buf306, 100352, 2048, stream=stream0)
        buf307 = reinterpret_tensor(buf296, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf296  # reuse
        buf308 = reinterpret_tensor(buf295, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf295  # reuse
        buf309 = reinterpret_tensor(buf294, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf304, buf305, buf306, buf307, buf308, buf309, 1024, 98, stream=stream0)
        buf310 = buf272; del buf272  # reuse
        buf311 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf313 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_67], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__32.run(buf307, buf308, buf309, primals_142, primals_143, buf310, buf311, buf313, primals_142, primals_143, 512, 2, stream=stream0)
        del primals_142
        del primals_143
        buf314 = empty_strided_cuda((512, 512, 28, 28), (401408, 1, 14336, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_67, out_68, out_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_35.run(buf303, buf310, buf311, primals_144, primals_145, buf275, buf314, 205520896, stream=stream0)
        del primals_145
        buf315 = empty_strided_cuda((256, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(primals_146, buf315, 131072, stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf314, buf315, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (512, 256, 28, 28), (200704, 1, 7168, 256), 'torch.ops.aten.convolution.default')
        buf317 = reinterpret_tensor(buf293, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf293  # reuse
        buf318 = reinterpret_tensor(buf292, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf292  # reuse
        buf319 = reinterpret_tensor(buf291, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf316, buf317, buf318, buf319, 131072, 784, stream=stream0)
        buf320 = reinterpret_tensor(buf309, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf309  # reuse
        buf321 = reinterpret_tensor(buf308, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf308  # reuse
        buf322 = reinterpret_tensor(buf307, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf317, buf318, buf319, buf320, buf321, buf322, 1024, 128, stream=stream0)
        buf323 = buf142; del buf142  # reuse
        buf324 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf326 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_71], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__38.run(buf320, buf321, buf322, primals_148, primals_149, buf323, buf324, buf326, primals_148, primals_149, 256, 4, stream=stream0)
        del primals_148
        del primals_149
        buf327 = empty_strided_cuda((512, 256, 28, 28), (200704, 1, 7168, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_39.run(buf316, buf323, buf324, primals_150, primals_151, buf327, 102760448, stream=stream0)
        del primals_151
        buf328 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_73], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(primals_152, buf328, 65536, 9, stream=stream0)
        del primals_152
        # Topologically Sorted Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf327, buf328, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf330 = buf319; del buf319  # reuse
        buf331 = buf318; del buf318  # reuse
        buf332 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf329, buf330, buf331, buf332, 131072, 196, stream=stream0)
        buf333 = buf322; del buf322  # reuse
        buf334 = buf321; del buf321  # reuse
        buf335 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf330, buf331, buf332, buf333, buf334, buf335, 1024, 128, stream=stream0)
        buf336 = buf324; del buf324  # reuse
        buf337 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf339 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf333, buf334, buf335, primals_154, primals_155, buf336, buf337, buf339, primals_154, primals_155, 256, 4, stream=stream0)
        del primals_154
        del primals_155
        buf340 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf329, buf336, buf337, primals_156, primals_157, buf340, 25690112, stream=stream0)
        del primals_157
        buf341 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_76], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_158, buf341, 262144, stream=stream0)
        del primals_158
        # Topologically Sorted Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf340, buf341, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf343 = reinterpret_tensor(buf332, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf332  # reuse
        buf344 = reinterpret_tensor(buf331, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf331  # reuse
        buf345 = reinterpret_tensor(buf330, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf342, buf343, buf344, buf345, 131072, 784, stream=stream0)
        buf346 = reinterpret_tensor(buf335, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf335  # reuse
        buf347 = reinterpret_tensor(buf334, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf334  # reuse
        buf349 = reinterpret_tensor(buf333, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_copy__46.run(buf343, buf344, buf345, primals_160, primals_161, buf346, buf347, buf349, primals_160, primals_161, 1024, 128, stream=stream0)
        del primals_160
        del primals_161
        buf350 = empty_strided_cuda((1024, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_47.run(primals_164, buf350, 524288, stream=stream0)
        del primals_164
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf314, buf350, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf352 = buf345; del buf345  # reuse
        buf353 = buf344; del buf344  # reuse
        buf354 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf351, buf352, buf353, buf354, 131072, 784, stream=stream0)
        buf355 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        buf356 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        buf358 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_copy__46.run(buf352, buf353, buf354, primals_166, primals_167, buf355, buf356, buf358, primals_166, primals_167, 1024, 128, stream=stream0)
        del primals_166
        del primals_167
        buf359 = empty_strided_cuda((512, 1024, 14, 14), (200704, 1, 14336, 1024), torch.bfloat16)
        buf360 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [out_77, input_6, out_78, out_79], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_48.run(buf360, buf342, buf346, buf347, primals_162, primals_163, buf351, buf355, buf356, primals_168, primals_169, 102760448, stream=stream0)
        del primals_163
        del primals_169
        buf361 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_80], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_170, buf361, 262144, stream=stream0)
        del primals_170
        # Topologically Sorted Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf360, buf361, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf363 = reinterpret_tensor(buf354, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf354  # reuse
        buf364 = reinterpret_tensor(buf353, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf353  # reuse
        buf365 = reinterpret_tensor(buf352, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf362, buf363, buf364, buf365, 131072, 196, stream=stream0)
        buf366 = reinterpret_tensor(buf356, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf356  # reuse
        buf367 = reinterpret_tensor(buf347, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf347  # reuse
        buf368 = empty_strided_cuda((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf363, buf364, buf365, buf366, buf367, buf368, 1024, 128, stream=stream0)
        buf369 = buf337; del buf337  # reuse
        buf370 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf372 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf366, buf367, buf368, primals_172, primals_173, buf369, buf370, buf372, primals_172, primals_173, 256, 4, stream=stream0)
        del primals_172
        del primals_173
        buf373 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf362, buf369, buf370, primals_174, primals_175, buf373, 25690112, stream=stream0)
        del primals_175
        buf374 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_83], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(primals_176, buf374, 65536, 9, stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf373, buf374, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf376 = buf365; del buf365  # reuse
        buf377 = buf364; del buf364  # reuse
        buf378 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [out_84], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf375, buf376, buf377, buf378, 131072, 196, stream=stream0)
        buf379 = buf368; del buf368  # reuse
        buf380 = buf367; del buf367  # reuse
        buf381 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [out_84], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf376, buf377, buf378, buf379, buf380, buf381, 1024, 128, stream=stream0)
        buf382 = buf370; del buf370  # reuse
        buf383 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf385 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_84], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf379, buf380, buf381, primals_178, primals_179, buf382, buf383, buf385, primals_178, primals_179, 256, 4, stream=stream0)
        del primals_178
        del primals_179
        buf386 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_84, out_85], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf375, buf382, buf383, primals_180, primals_181, buf386, 25690112, stream=stream0)
        del primals_181
        buf387 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_86], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_182, buf387, 262144, stream=stream0)
        del primals_182
        # Topologically Sorted Source Nodes: [out_86], Original ATen: [aten.convolution]
        buf388 = extern_kernels.convolution(buf386, buf387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf389 = reinterpret_tensor(buf378, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf378  # reuse
        buf390 = reinterpret_tensor(buf377, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf377  # reuse
        buf391 = reinterpret_tensor(buf376, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf376  # reuse
        # Topologically Sorted Source Nodes: [out_87], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf388, buf389, buf390, buf391, 131072, 784, stream=stream0)
        buf392 = reinterpret_tensor(buf381, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf381  # reuse
        buf393 = reinterpret_tensor(buf380, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf380  # reuse
        buf395 = reinterpret_tensor(buf379, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf379  # reuse
        # Topologically Sorted Source Nodes: [out_87], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_copy__46.run(buf389, buf390, buf391, primals_184, primals_185, buf392, buf393, buf395, primals_184, primals_185, 1024, 128, stream=stream0)
        del primals_184
        del primals_185
        buf396 = empty_strided_cuda((512, 1024, 14, 14), (200704, 1, 14336, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_87, out_88, out_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf388, buf392, buf393, primals_186, primals_187, buf360, buf396, 102760448, stream=stream0)
        del primals_187
        buf397 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_90], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_188, buf397, 262144, stream=stream0)
        del primals_188
        # Topologically Sorted Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf398 = extern_kernels.convolution(buf396, buf397, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf399 = reinterpret_tensor(buf391, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf391  # reuse
        buf400 = reinterpret_tensor(buf390, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf390  # reuse
        buf401 = reinterpret_tensor(buf389, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [out_91], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf398, buf399, buf400, buf401, 131072, 196, stream=stream0)
        buf402 = reinterpret_tensor(buf393, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf393  # reuse
        buf403 = empty_strided_cuda((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), torch.float32)
        buf404 = empty_strided_cuda((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_91], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf399, buf400, buf401, buf402, buf403, buf404, 1024, 128, stream=stream0)
        buf405 = buf383; del buf383  # reuse
        buf406 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf408 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_91], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf402, buf403, buf404, primals_190, primals_191, buf405, buf406, buf408, primals_190, primals_191, 256, 4, stream=stream0)
        del primals_190
        del primals_191
        buf409 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_91, out_92], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf398, buf405, buf406, primals_192, primals_193, buf409, 25690112, stream=stream0)
        del primals_193
        buf410 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_93], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(primals_194, buf410, 65536, 9, stream=stream0)
        del primals_194
        # Topologically Sorted Source Nodes: [out_93], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf409, buf410, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf412 = buf401; del buf401  # reuse
        buf413 = buf400; del buf400  # reuse
        buf414 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [out_94], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf411, buf412, buf413, buf414, 131072, 196, stream=stream0)
        buf415 = buf404; del buf404  # reuse
        buf416 = buf403; del buf403  # reuse
        buf417 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [out_94], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf412, buf413, buf414, buf415, buf416, buf417, 1024, 128, stream=stream0)
        buf418 = buf406; del buf406  # reuse
        buf419 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf421 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf415, buf416, buf417, primals_196, primals_197, buf418, buf419, buf421, primals_196, primals_197, 256, 4, stream=stream0)
        del primals_196
        del primals_197
        buf422 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_94, out_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf411, buf418, buf419, primals_198, primals_199, buf422, 25690112, stream=stream0)
        del primals_199
        buf423 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_96], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_200, buf423, 262144, stream=stream0)
        del primals_200
        # Topologically Sorted Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf422, buf423, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf425 = reinterpret_tensor(buf414, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf414  # reuse
        buf426 = reinterpret_tensor(buf413, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf413  # reuse
        buf427 = reinterpret_tensor(buf412, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf424, buf425, buf426, buf427, 131072, 784, stream=stream0)
        buf428 = reinterpret_tensor(buf417, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf417  # reuse
        buf429 = reinterpret_tensor(buf416, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf416  # reuse
        buf431 = reinterpret_tensor(buf415, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_copy__46.run(buf425, buf426, buf427, primals_202, primals_203, buf428, buf429, buf431, primals_202, primals_203, 1024, 128, stream=stream0)
        del primals_202
        del primals_203
        buf432 = empty_strided_cuda((512, 1024, 14, 14), (200704, 1, 14336, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_97, out_98, out_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf424, buf428, buf429, primals_204, primals_205, buf396, buf432, 102760448, stream=stream0)
        del primals_205
        buf433 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_100], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_206, buf433, 262144, stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf432, buf433, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf435 = reinterpret_tensor(buf427, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf427  # reuse
        buf436 = reinterpret_tensor(buf426, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf426  # reuse
        buf437 = reinterpret_tensor(buf425, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf425  # reuse
        # Topologically Sorted Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf434, buf435, buf436, buf437, 131072, 196, stream=stream0)
        buf438 = reinterpret_tensor(buf429, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf429  # reuse
        buf439 = empty_strided_cuda((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), torch.float32)
        buf440 = empty_strided_cuda((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf435, buf436, buf437, buf438, buf439, buf440, 1024, 128, stream=stream0)
        buf441 = buf419; del buf419  # reuse
        buf442 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf444 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf438, buf439, buf440, primals_208, primals_209, buf441, buf442, buf444, primals_208, primals_209, 256, 4, stream=stream0)
        del primals_208
        del primals_209
        buf445 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf434, buf441, buf442, primals_210, primals_211, buf445, 25690112, stream=stream0)
        del primals_211
        buf446 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_103], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(primals_212, buf446, 65536, 9, stream=stream0)
        del primals_212
        # Topologically Sorted Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf445, buf446, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf448 = buf437; del buf437  # reuse
        buf449 = buf436; del buf436  # reuse
        buf450 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf447, buf448, buf449, buf450, 131072, 196, stream=stream0)
        buf451 = buf440; del buf440  # reuse
        buf452 = buf439; del buf439  # reuse
        buf453 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf448, buf449, buf450, buf451, buf452, buf453, 1024, 128, stream=stream0)
        buf454 = buf442; del buf442  # reuse
        buf455 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf457 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_104], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf451, buf452, buf453, primals_214, primals_215, buf454, buf455, buf457, primals_214, primals_215, 256, 4, stream=stream0)
        del primals_214
        del primals_215
        buf458 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf447, buf454, buf455, primals_216, primals_217, buf458, 25690112, stream=stream0)
        del primals_217
        buf459 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_218, buf459, 262144, stream=stream0)
        del primals_218
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf460 = extern_kernels.convolution(buf458, buf459, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf460, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf461 = reinterpret_tensor(buf450, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf450  # reuse
        buf462 = reinterpret_tensor(buf449, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf449  # reuse
        buf463 = reinterpret_tensor(buf448, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [out_107], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf460, buf461, buf462, buf463, 131072, 784, stream=stream0)
        buf464 = reinterpret_tensor(buf453, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf453  # reuse
        buf465 = reinterpret_tensor(buf452, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf452  # reuse
        buf467 = reinterpret_tensor(buf451, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [out_107], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_copy__46.run(buf461, buf462, buf463, primals_220, primals_221, buf464, buf465, buf467, primals_220, primals_221, 1024, 128, stream=stream0)
        del primals_220
        del primals_221
        buf468 = empty_strided_cuda((512, 1024, 14, 14), (200704, 1, 14336, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_107, out_108, out_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf460, buf464, buf465, primals_222, primals_223, buf432, buf468, 102760448, stream=stream0)
        del primals_223
        buf469 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_224, buf469, 262144, stream=stream0)
        del primals_224
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf468, buf469, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf471 = reinterpret_tensor(buf463, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf463  # reuse
        buf472 = reinterpret_tensor(buf462, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf462  # reuse
        buf473 = reinterpret_tensor(buf461, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf461  # reuse
        # Topologically Sorted Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf470, buf471, buf472, buf473, 131072, 196, stream=stream0)
        buf474 = reinterpret_tensor(buf465, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf465  # reuse
        buf475 = empty_strided_cuda((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), torch.float32)
        buf476 = empty_strided_cuda((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf471, buf472, buf473, buf474, buf475, buf476, 1024, 128, stream=stream0)
        buf477 = buf455; del buf455  # reuse
        buf478 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf480 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf474, buf475, buf476, primals_226, primals_227, buf477, buf478, buf480, primals_226, primals_227, 256, 4, stream=stream0)
        del primals_226
        del primals_227
        buf481 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf470, buf477, buf478, primals_228, primals_229, buf481, 25690112, stream=stream0)
        del primals_229
        buf482 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(primals_230, buf482, 65536, 9, stream=stream0)
        del primals_230
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf481, buf482, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf484 = buf473; del buf473  # reuse
        buf485 = buf472; del buf472  # reuse
        buf486 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [out_114], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf483, buf484, buf485, buf486, 131072, 196, stream=stream0)
        buf487 = buf476; del buf476  # reuse
        buf488 = buf475; del buf475  # reuse
        buf489 = buf474; del buf474  # reuse
        # Topologically Sorted Source Nodes: [out_114], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf484, buf485, buf486, buf487, buf488, buf489, 1024, 128, stream=stream0)
        buf490 = buf478; del buf478  # reuse
        buf491 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf493 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf487, buf488, buf489, primals_232, primals_233, buf490, buf491, buf493, primals_232, primals_233, 256, 4, stream=stream0)
        del primals_232
        del primals_233
        buf494 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_114, out_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf483, buf490, buf491, primals_234, primals_235, buf494, 25690112, stream=stream0)
        del primals_235
        buf495 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_116], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_236, buf495, 262144, stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf496 = extern_kernels.convolution(buf494, buf495, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf497 = reinterpret_tensor(buf486, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf486  # reuse
        buf498 = reinterpret_tensor(buf485, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf485  # reuse
        buf499 = reinterpret_tensor(buf484, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf484  # reuse
        # Topologically Sorted Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf496, buf497, buf498, buf499, 131072, 784, stream=stream0)
        buf500 = reinterpret_tensor(buf489, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf489  # reuse
        buf501 = reinterpret_tensor(buf488, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf488  # reuse
        buf503 = reinterpret_tensor(buf487, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_copy__46.run(buf497, buf498, buf499, primals_238, primals_239, buf500, buf501, buf503, primals_238, primals_239, 1024, 128, stream=stream0)
        del primals_238
        del primals_239
        buf504 = empty_strided_cuda((512, 1024, 14, 14), (200704, 1, 14336, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_117, out_118, out_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf496, buf500, buf501, primals_240, primals_241, buf468, buf504, 102760448, stream=stream0)
        del primals_241
        buf505 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_120], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_242, buf505, 262144, stream=stream0)
        del primals_242
        # Topologically Sorted Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf504, buf505, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf507 = reinterpret_tensor(buf499, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf499  # reuse
        buf508 = reinterpret_tensor(buf498, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf498  # reuse
        buf509 = reinterpret_tensor(buf497, (1, 256, 1, 1, 512), (131072, 1, 131072, 131072, 256), 0); del buf497  # reuse
        # Topologically Sorted Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf506, buf507, buf508, buf509, 131072, 196, stream=stream0)
        buf510 = reinterpret_tensor(buf501, (1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), 0); del buf501  # reuse
        buf511 = empty_strided_cuda((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), torch.float32)
        buf512 = empty_strided_cuda((1, 256, 1, 1, 4), (1024, 1, 1024, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf507, buf508, buf509, buf510, buf511, buf512, 1024, 128, stream=stream0)
        buf513 = buf491; del buf491  # reuse
        buf514 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf516 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf510, buf511, buf512, primals_244, primals_245, buf513, buf514, buf516, primals_244, primals_245, 256, 4, stream=stream0)
        del primals_244
        del primals_245
        buf517 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf506, buf513, buf514, primals_246, primals_247, buf517, 25690112, stream=stream0)
        del primals_247
        buf518 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_123], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_40.run(primals_248, buf518, 65536, 9, stream=stream0)
        del primals_248
        # Topologically Sorted Source Nodes: [out_123], Original ATen: [aten.convolution]
        buf519 = extern_kernels.convolution(buf517, buf518, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf519, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        buf520 = buf509; del buf509  # reuse
        buf521 = buf508; del buf508  # reuse
        buf522 = buf507; del buf507  # reuse
        # Topologically Sorted Source Nodes: [out_124], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf519, buf520, buf521, buf522, 131072, 196, stream=stream0)
        buf523 = buf512; del buf512  # reuse
        buf524 = buf511; del buf511  # reuse
        buf525 = buf510; del buf510  # reuse
        # Topologically Sorted Source Nodes: [out_124], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf520, buf521, buf522, buf523, buf524, buf525, 1024, 128, stream=stream0)
        buf526 = buf514; del buf514  # reuse
        buf527 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf529 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__42.run(buf523, buf524, buf525, primals_250, primals_251, buf526, buf527, buf529, primals_250, primals_251, 256, 4, stream=stream0)
        del primals_250
        del primals_251
        buf530 = empty_strided_cuda((512, 256, 14, 14), (50176, 1, 3584, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_124, out_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_43.run(buf519, buf526, buf527, primals_252, primals_253, buf530, 25690112, stream=stream0)
        del buf527
        del primals_253
        buf531 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_126], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_44.run(primals_254, buf531, 262144, stream=stream0)
        del primals_254
        # Topologically Sorted Source Nodes: [out_126], Original ATen: [aten.convolution]
        buf532 = extern_kernels.convolution(buf530, buf531, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf532, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        buf533 = reinterpret_tensor(buf522, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf522  # reuse
        buf534 = reinterpret_tensor(buf521, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf521  # reuse
        buf535 = reinterpret_tensor(buf520, (1, 1024, 1, 1, 128), (131072, 1, 131072, 131072, 1024), 0); del buf520  # reuse
        # Topologically Sorted Source Nodes: [out_127], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf532, buf533, buf534, buf535, 131072, 784, stream=stream0)
        buf536 = reinterpret_tensor(buf525, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf525  # reuse
        buf537 = reinterpret_tensor(buf524, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf524  # reuse
        buf539 = reinterpret_tensor(buf523, (1, 1024, 1, 1), (1024, 1, 1024, 1024), 0); del buf523  # reuse
        # Topologically Sorted Source Nodes: [out_127], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_copy__46.run(buf533, buf534, buf535, primals_256, primals_257, buf536, buf537, buf539, primals_256, primals_257, 1024, 128, stream=stream0)
        del primals_256
        del primals_257
        buf540 = empty_strided_cuda((512, 1024, 14, 14), (200704, 1, 14336, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_127, out_128, out_129], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_49.run(buf532, buf536, buf537, primals_258, primals_259, buf504, buf540, 102760448, stream=stream0)
        del primals_259
        buf541 = empty_strided_cuda((512, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_130], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_47.run(primals_260, buf541, 524288, stream=stream0)
        del primals_260
        # Topologically Sorted Source Nodes: [out_130], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf540, buf541, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (512, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution.default')
        buf543 = reinterpret_tensor(buf535, (1, 512, 1, 1, 256), (131072, 1, 131072, 131072, 512), 0); del buf535  # reuse
        buf544 = reinterpret_tensor(buf534, (1, 512, 1, 1, 256), (131072, 1, 131072, 131072, 512), 0); del buf534  # reuse
        buf545 = reinterpret_tensor(buf533, (1, 512, 1, 1, 256), (131072, 1, 131072, 131072, 512), 0); del buf533  # reuse
        # Topologically Sorted Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_50.run(buf542, buf543, buf544, buf545, 131072, 392, stream=stream0)
        buf546 = reinterpret_tensor(buf537, (1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), 0); del buf537  # reuse
        buf547 = empty_strided_cuda((1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), torch.float32)
        buf548 = empty_strided_cuda((1, 512, 1, 1, 2), (1024, 1, 1024, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf543, buf544, buf545, buf546, buf547, buf548, 1024, 128, stream=stream0)
        buf549 = buf311; del buf311  # reuse
        buf550 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf552 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__52.run(buf546, buf547, buf548, primals_262, primals_263, buf549, buf550, buf552, primals_262, primals_263, 512, 2, stream=stream0)
        del primals_262
        del primals_263
        buf553 = empty_strided_cuda((512, 512, 14, 14), (100352, 1, 7168, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_53.run(buf542, buf549, buf550, primals_264, primals_265, buf553, 51380224, stream=stream0)
        del primals_265
        buf554 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_133], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(primals_266, buf554, 262144, 9, stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [out_133], Original ATen: [aten.convolution]
        buf555 = extern_kernels.convolution(buf553, buf554, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf555, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf556 = buf306; del buf306  # reuse
        buf557 = buf305; del buf305  # reuse
        buf558 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf555, buf556, buf557, buf558, 100352, 128, stream=stream0)
        buf559 = buf548; del buf548  # reuse
        buf560 = buf547; del buf547  # reuse
        buf561 = buf546; del buf546  # reuse
        # Topologically Sorted Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf556, buf557, buf558, buf559, buf560, buf561, 1024, 98, stream=stream0)
        buf562 = buf550; del buf550  # reuse
        buf563 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf565 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_134], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__56.run(buf559, buf560, buf561, primals_268, primals_269, buf562, buf563, buf565, primals_268, primals_269, 512, 2, stream=stream0)
        del primals_268
        del primals_269
        buf566 = empty_strided_cuda((512, 512, 7, 7), (25088, 1, 3584, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf555, buf562, buf563, primals_270, primals_271, buf566, 12845056, stream=stream0)
        del primals_271
        buf567 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_136], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_58.run(primals_272, buf567, 1048576, stream=stream0)
        del primals_272
        # Topologically Sorted Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf568 = extern_kernels.convolution(buf566, buf567, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf568, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        buf569 = reinterpret_tensor(buf545, (1, 2048, 1, 1, 64), (131072, 1, 131072, 131072, 2048), 0); del buf545  # reuse
        buf570 = reinterpret_tensor(buf544, (1, 2048, 1, 1, 64), (131072, 1, 131072, 131072, 2048), 0); del buf544  # reuse
        buf571 = reinterpret_tensor(buf543, (1, 2048, 1, 1, 64), (131072, 1, 131072, 131072, 2048), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf568, buf569, buf570, buf571, 131072, 392, stream=stream0)
        buf572 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        buf573 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        buf575 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__60.run(buf569, buf570, buf571, primals_274, primals_275, buf572, buf573, buf575, primals_274, primals_275, 2048, 64, stream=stream0)
        del primals_274
        del primals_275
        buf576 = empty_strided_cuda((2048, 1024, 1, 1), (1024, 1, 1024, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_61.run(primals_278, buf576, 2097152, stream=stream0)
        del primals_278
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf577 = extern_kernels.convolution(buf540, buf576, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf577, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        buf578 = buf571; del buf571  # reuse
        buf579 = buf570; del buf570  # reuse
        buf580 = buf569; del buf569  # reuse
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf577, buf578, buf579, buf580, 131072, 392, stream=stream0)
        buf581 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        buf582 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        buf584 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__60.run(buf578, buf579, buf580, primals_280, primals_281, buf581, buf582, buf584, primals_280, primals_281, 2048, 64, stream=stream0)
        del primals_280
        del primals_281
        buf585 = empty_strided_cuda((512, 2048, 7, 7), (100352, 1, 14336, 2048), torch.bfloat16)
        buf586 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [out_137, input_8, out_138, out_139], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_62.run(buf586, buf568, buf572, buf573, primals_276, primals_277, buf577, buf581, buf582, primals_282, primals_283, 51380224, stream=stream0)
        del primals_277
        del primals_283
        buf587 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_58.run(primals_284, buf587, 1048576, stream=stream0)
        del primals_284
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf588 = extern_kernels.convolution(buf586, buf587, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf588, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf589 = buf558; del buf558  # reuse
        buf590 = buf557; del buf557  # reuse
        buf591 = buf556; del buf556  # reuse
        # Topologically Sorted Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf588, buf589, buf590, buf591, 100352, 128, stream=stream0)
        buf592 = buf561; del buf561  # reuse
        buf593 = buf560; del buf560  # reuse
        buf594 = buf559; del buf559  # reuse
        # Topologically Sorted Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf589, buf590, buf591, buf592, buf593, buf594, 1024, 98, stream=stream0)
        buf595 = buf563; del buf563  # reuse
        buf596 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf598 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_141], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__56.run(buf592, buf593, buf594, primals_286, primals_287, buf595, buf596, buf598, primals_286, primals_287, 512, 2, stream=stream0)
        del primals_286
        del primals_287
        buf599 = empty_strided_cuda((512, 512, 7, 7), (25088, 1, 3584, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_141, out_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf588, buf595, buf596, primals_288, primals_289, buf599, 12845056, stream=stream0)
        del primals_289
        buf600 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_143], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(primals_290, buf600, 262144, 9, stream=stream0)
        del primals_290
        # Topologically Sorted Source Nodes: [out_143], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(buf599, buf600, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf601, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf602 = buf591; del buf591  # reuse
        buf603 = buf590; del buf590  # reuse
        buf604 = buf589; del buf589  # reuse
        # Topologically Sorted Source Nodes: [out_144], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf601, buf602, buf603, buf604, 100352, 128, stream=stream0)
        buf605 = buf594; del buf594  # reuse
        buf606 = buf593; del buf593  # reuse
        buf607 = buf592; del buf592  # reuse
        # Topologically Sorted Source Nodes: [out_144], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf602, buf603, buf604, buf605, buf606, buf607, 1024, 98, stream=stream0)
        buf608 = buf596; del buf596  # reuse
        buf609 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf611 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_144], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__56.run(buf605, buf606, buf607, primals_292, primals_293, buf608, buf609, buf611, primals_292, primals_293, 512, 2, stream=stream0)
        del primals_292
        del primals_293
        buf612 = empty_strided_cuda((512, 512, 7, 7), (25088, 1, 3584, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_144, out_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf601, buf608, buf609, primals_294, primals_295, buf612, 12845056, stream=stream0)
        del primals_295
        buf613 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_146], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_58.run(primals_296, buf613, 1048576, stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [out_146], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf612, buf613, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        buf615 = buf580; del buf580  # reuse
        buf616 = buf579; del buf579  # reuse
        buf617 = buf578; del buf578  # reuse
        # Topologically Sorted Source Nodes: [out_147], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf614, buf615, buf616, buf617, 131072, 392, stream=stream0)
        buf618 = buf582; del buf582  # reuse
        buf619 = buf573; del buf573  # reuse
        buf621 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [out_147], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__60.run(buf615, buf616, buf617, primals_298, primals_299, buf618, buf619, buf621, primals_298, primals_299, 2048, 64, stream=stream0)
        del primals_298
        del primals_299
        buf622 = empty_strided_cuda((512, 2048, 7, 7), (100352, 1, 14336, 2048), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_147, out_148, out_149], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_63.run(buf614, buf618, buf619, primals_300, primals_301, buf586, buf622, 51380224, stream=stream0)
        del primals_301
        buf623 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_150], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_58.run(primals_302, buf623, 1048576, stream=stream0)
        del primals_302
        # Topologically Sorted Source Nodes: [out_150], Original ATen: [aten.convolution]
        buf624 = extern_kernels.convolution(buf622, buf623, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf624, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf625 = buf604; del buf604  # reuse
        buf626 = buf603; del buf603  # reuse
        buf627 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [out_151], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf624, buf625, buf626, buf627, 100352, 128, stream=stream0)
        buf628 = buf607; del buf607  # reuse
        buf629 = buf606; del buf606  # reuse
        buf630 = buf605; del buf605  # reuse
        # Topologically Sorted Source Nodes: [out_151], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf625, buf626, buf627, buf628, buf629, buf630, 1024, 98, stream=stream0)
        buf631 = buf609; del buf609  # reuse
        buf632 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf634 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_151], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__56.run(buf628, buf629, buf630, primals_304, primals_305, buf631, buf632, buf634, primals_304, primals_305, 512, 2, stream=stream0)
        del primals_304
        del primals_305
        buf635 = empty_strided_cuda((512, 512, 7, 7), (25088, 1, 3584, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_151, out_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf624, buf631, buf632, primals_306, primals_307, buf635, 12845056, stream=stream0)
        del primals_307
        buf636 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_153], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_54.run(primals_308, buf636, 262144, 9, stream=stream0)
        del primals_308
        # Topologically Sorted Source Nodes: [out_153], Original ATen: [aten.convolution]
        buf637 = extern_kernels.convolution(buf635, buf636, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf637, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        buf638 = buf627; del buf627  # reuse
        buf639 = buf626; del buf626  # reuse
        buf640 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [out_154], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf637, buf638, buf639, buf640, 100352, 128, stream=stream0)
        buf641 = buf630; del buf630  # reuse
        buf642 = buf629; del buf629  # reuse
        buf643 = buf628; del buf628  # reuse
        # Topologically Sorted Source Nodes: [out_154], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf638, buf639, buf640, buf641, buf642, buf643, 1024, 98, stream=stream0)
        del buf638
        del buf639
        del buf640
        buf644 = buf632; del buf632  # reuse
        buf645 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf647 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__56.run(buf641, buf642, buf643, primals_310, primals_311, buf644, buf645, buf647, primals_310, primals_311, 512, 2, stream=stream0)
        del buf641
        del buf642
        del buf643
        del primals_310
        del primals_311
        buf648 = empty_strided_cuda((512, 512, 7, 7), (25088, 1, 3584, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_154, out_155], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf637, buf644, buf645, primals_312, primals_313, buf648, 12845056, stream=stream0)
        del buf645
        del primals_313
        buf649 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_156], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_58.run(primals_314, buf649, 1048576, stream=stream0)
        del primals_314
        # Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf650 = extern_kernels.convolution(buf648, buf649, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf650, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        buf651 = buf617; del buf617  # reuse
        buf652 = buf616; del buf616  # reuse
        buf653 = buf615; del buf615  # reuse
        # Topologically Sorted Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_59.run(buf650, buf651, buf652, buf653, 131072, 392, stream=stream0)
        buf654 = buf619; del buf619  # reuse
        buf655 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        buf657 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [out_157], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_copy__60.run(buf651, buf652, buf653, primals_316, primals_317, buf654, buf655, buf657, primals_316, primals_317, 2048, 64, stream=stream0)
        del buf651
        del buf652
        del buf653
        del primals_316
        del primals_317
        buf658 = empty_strided_cuda((512, 2048, 7, 7), (100352, 1, 14336, 2048), torch.bfloat16)
        buf664 = empty_strided_cuda((512, 2048, 7, 7), (100352, 1, 14336, 2048), torch.bool)
        # Topologically Sorted Source Nodes: [out_157, out_158, out_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_64.run(buf650, buf654, buf655, primals_318, primals_319, buf622, buf658, buf664, 51380224, stream=stream0)
        del buf655
        del primals_319
        buf660 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 1048576, 1048576), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_65.run(buf658, buf660, 1048576, 49, stream=stream0)
        del buf658
        buf661 = empty_strided_cuda((100, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_66.run(primals_320, buf661, 204800, stream=stream0)
        del primals_320
        buf662 = empty_strided_cuda((100, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_67.run(primals_321, buf662, 100, stream=stream0)
        del primals_321
        buf663 = empty_strided_cuda((512, 100), (100, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf662, reinterpret_tensor(buf660, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf661, (2048, 100), (1, 2048), 0), alpha=1, beta=1, out=buf663)
        del buf662
        # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_3, primals_3, 1, stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_9, primals_9, 1, stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [add__2], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_15, primals_15, 1, stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [add__3], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_21, primals_21, 1, stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [add__4], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_27, primals_27, 1, stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [add__5], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_33, primals_33, 1, stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [add__6], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_39, primals_39, 1, stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [add__7], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_45, primals_45, 1, stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [add__8], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_51, primals_51, 1, stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [add__9], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_57, primals_57, 1, stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [add__10], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_63, primals_63, 1, stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [add__11], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_69, primals_69, 1, stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [add__12], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_75, primals_75, 1, stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [add__13], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_81, primals_81, 1, stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [add__14], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_87, primals_87, 1, stream=stream0)
        del primals_87
        # Topologically Sorted Source Nodes: [add__15], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_93, primals_93, 1, stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [add__16], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_99, primals_99, 1, stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [add__17], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_105, primals_105, 1, stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [add__18], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_111, primals_111, 1, stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [add__19], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_117, primals_117, 1, stream=stream0)
        del primals_117
        # Topologically Sorted Source Nodes: [add__20], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_123, primals_123, 1, stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [add__21], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_129, primals_129, 1, stream=stream0)
        del primals_129
        # Topologically Sorted Source Nodes: [add__22], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_135, primals_135, 1, stream=stream0)
        del primals_135
        # Topologically Sorted Source Nodes: [add__23], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_141, primals_141, 1, stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [add__24], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_147, primals_147, 1, stream=stream0)
        del primals_147
        # Topologically Sorted Source Nodes: [add__25], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_153, primals_153, 1, stream=stream0)
        del primals_153
        # Topologically Sorted Source Nodes: [add__26], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_159, primals_159, 1, stream=stream0)
        del primals_159
        # Topologically Sorted Source Nodes: [add__27], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_165, primals_165, 1, stream=stream0)
        del primals_165
        # Topologically Sorted Source Nodes: [add__28], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_171, primals_171, 1, stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [add__29], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_177, primals_177, 1, stream=stream0)
        del primals_177
        # Topologically Sorted Source Nodes: [add__30], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_183, primals_183, 1, stream=stream0)
        del primals_183
        # Topologically Sorted Source Nodes: [add__31], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_189, primals_189, 1, stream=stream0)
        del primals_189
        # Topologically Sorted Source Nodes: [add__32], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_195, primals_195, 1, stream=stream0)
        del primals_195
        # Topologically Sorted Source Nodes: [add__33], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_201, primals_201, 1, stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [add__34], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_207, primals_207, 1, stream=stream0)
        del primals_207
        # Topologically Sorted Source Nodes: [add__35], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_213, primals_213, 1, stream=stream0)
        del primals_213
        # Topologically Sorted Source Nodes: [add__36], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_219, primals_219, 1, stream=stream0)
        del primals_219
        # Topologically Sorted Source Nodes: [add__37], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_225, primals_225, 1, stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [add__38], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_231, primals_231, 1, stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [add__39], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_237, primals_237, 1, stream=stream0)
        del primals_237
        # Topologically Sorted Source Nodes: [add__40], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_243, primals_243, 1, stream=stream0)
        del primals_243
        # Topologically Sorted Source Nodes: [add__41], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_249, primals_249, 1, stream=stream0)
        del primals_249
        # Topologically Sorted Source Nodes: [add__42], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_255, primals_255, 1, stream=stream0)
        del primals_255
        # Topologically Sorted Source Nodes: [add__43], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_261, primals_261, 1, stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [add__44], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_267, primals_267, 1, stream=stream0)
        del primals_267
        # Topologically Sorted Source Nodes: [add__45], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_273, primals_273, 1, stream=stream0)
        del primals_273
        # Topologically Sorted Source Nodes: [add__46], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_279, primals_279, 1, stream=stream0)
        del primals_279
        # Topologically Sorted Source Nodes: [add__47], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_285, primals_285, 1, stream=stream0)
        del primals_285
        # Topologically Sorted Source Nodes: [add__48], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_291, primals_291, 1, stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [add__49], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_297, primals_297, 1, stream=stream0)
        del primals_297
        # Topologically Sorted Source Nodes: [add__50], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_303, primals_303, 1, stream=stream0)
        del primals_303
        # Topologically Sorted Source Nodes: [add__51], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_309, primals_309, 1, stream=stream0)
        del primals_309
        # Topologically Sorted Source Nodes: [add__52], Original ATen: [aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_copy__68.run(primals_315, primals_315, 1, stream=stream0)
        del primals_315
    return (buf663, primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, primals_132, primals_138, primals_144, primals_150, primals_156, primals_162, primals_168, primals_174, primals_180, primals_186, primals_192, primals_198, primals_204, primals_210, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, primals_252, primals_258, primals_264, primals_270, primals_276, primals_282, primals_288, primals_294, primals_300, primals_306, primals_312, primals_318, buf0, buf1, buf2, reinterpret_tensor(buf12, (64, ), (1, ), 0), buf13, buf14, buf15, buf16, buf17, reinterpret_tensor(buf27, (64, ), (1, ), 0), buf28, buf29, buf30, reinterpret_tensor(buf40, (64, ), (1, ), 0), buf41, buf42, buf43, reinterpret_tensor(buf53, (256, ), (1, ), 0), buf54, buf55, reinterpret_tensor(buf65, (256, ), (1, ), 0), buf67, buf68, buf69, reinterpret_tensor(buf79, (64, ), (1, ), 0), buf80, buf81, buf82, reinterpret_tensor(buf92, (64, ), (1, ), 0), buf93, buf94, buf95, reinterpret_tensor(buf105, (256, ), (1, ), 0), buf106, buf107, buf108, reinterpret_tensor(buf118, (64, ), (1, ), 0), buf119, buf120, buf121, reinterpret_tensor(buf131, (64, ), (1, ), 0), buf132, buf133, buf134, reinterpret_tensor(buf144, (256, ), (1, ), 0), buf145, buf146, buf147, reinterpret_tensor(buf157, (128, ), (1, ), 0), buf158, buf159, buf160, reinterpret_tensor(buf170, (128, ), (1, ), 0), buf171, buf172, buf173, reinterpret_tensor(buf183, (512, ), (1, ), 0), buf184, buf185, reinterpret_tensor(buf195, (512, ), (1, ), 0), buf197, buf198, buf199, reinterpret_tensor(buf209, (128, ), (1, ), 0), buf210, buf211, buf212, reinterpret_tensor(buf222, (128, ), (1, ), 0), buf223, buf224, buf225, reinterpret_tensor(buf235, (512, ), (1, ), 0), buf236, buf237, buf238, reinterpret_tensor(buf248, (128, ), (1, ), 0), buf249, buf250, buf251, reinterpret_tensor(buf261, (128, ), (1, ), 0), buf262, buf263, buf264, reinterpret_tensor(buf274, (512, ), (1, ), 0), buf275, buf276, buf277, reinterpret_tensor(buf287, (128, ), (1, ), 0), buf288, buf289, buf290, reinterpret_tensor(buf300, (128, ), (1, ), 0), buf301, buf302, buf303, reinterpret_tensor(buf313, (512, ), (1, ), 0), buf314, buf315, buf316, reinterpret_tensor(buf326, (256, ), (1, ), 0), buf327, buf328, buf329, reinterpret_tensor(buf339, (256, ), (1, ), 0), buf340, buf341, buf342, reinterpret_tensor(buf349, (1024, ), (1, ), 0), buf350, buf351, reinterpret_tensor(buf358, (1024, ), (1, ), 0), buf360, buf361, buf362, reinterpret_tensor(buf372, (256, ), (1, ), 0), buf373, buf374, buf375, reinterpret_tensor(buf385, (256, ), (1, ), 0), buf386, buf387, buf388, reinterpret_tensor(buf395, (1024, ), (1, ), 0), buf396, buf397, buf398, reinterpret_tensor(buf408, (256, ), (1, ), 0), buf409, buf410, buf411, reinterpret_tensor(buf421, (256, ), (1, ), 0), buf422, buf423, buf424, reinterpret_tensor(buf431, (1024, ), (1, ), 0), buf432, buf433, buf434, reinterpret_tensor(buf444, (256, ), (1, ), 0), buf445, buf446, buf447, reinterpret_tensor(buf457, (256, ), (1, ), 0), buf458, buf459, buf460, reinterpret_tensor(buf467, (1024, ), (1, ), 0), buf468, buf469, buf470, reinterpret_tensor(buf480, (256, ), (1, ), 0), buf481, buf482, buf483, reinterpret_tensor(buf493, (256, ), (1, ), 0), buf494, buf495, buf496, reinterpret_tensor(buf503, (1024, ), (1, ), 0), buf504, buf505, buf506, reinterpret_tensor(buf516, (256, ), (1, ), 0), buf517, buf518, buf519, reinterpret_tensor(buf529, (256, ), (1, ), 0), buf530, buf531, buf532, reinterpret_tensor(buf539, (1024, ), (1, ), 0), buf540, buf541, buf542, reinterpret_tensor(buf552, (512, ), (1, ), 0), buf553, buf554, buf555, reinterpret_tensor(buf565, (512, ), (1, ), 0), buf566, buf567, buf568, reinterpret_tensor(buf575, (2048, ), (1, ), 0), buf576, buf577, reinterpret_tensor(buf584, (2048, ), (1, ), 0), buf586, buf587, buf588, reinterpret_tensor(buf598, (512, ), (1, ), 0), buf599, buf600, buf601, reinterpret_tensor(buf611, (512, ), (1, ), 0), buf612, buf613, buf614, reinterpret_tensor(buf621, (2048, ), (1, ), 0), buf622, buf623, buf624, reinterpret_tensor(buf634, (512, ), (1, ), 0), buf635, buf636, buf637, reinterpret_tensor(buf647, (512, ), (1, ), 0), buf648, buf649, buf650, reinterpret_tensor(buf657, (2048, ), (1, ), 0), reinterpret_tensor(buf660, (512, 2048), (2048, 1), 0), buf661, buf664, reinterpret_tensor(buf654, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf644, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf631, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf618, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf608, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf595, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf581, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf572, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf562, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf549, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf536, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf526, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf513, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf500, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf490, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf477, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf464, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf454, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf441, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf428, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf418, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf405, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf392, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf382, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf369, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf355, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf346, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf336, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf323, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf310, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf297, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf284, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf271, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf258, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf245, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf232, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf219, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf206, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf192, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf180, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf167, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf154, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf141, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf128, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf115, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf102, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf89, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf76, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf62, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf50, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf37, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf9, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


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
    primals_321 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
