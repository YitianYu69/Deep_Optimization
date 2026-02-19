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


# kernel path: /tmp/torchinductor_yyu496/i5/ci5gyi7wajk2nvyuhnogxmezz3atko3p3zy55irutvkp2vo4j52j.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_157], Original ATen: [aten.div, aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_157 => convert_element_type_158
#   scalar_tensor => full_default
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %div), kwargs = {})
#   %convert_element_type_172 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %sum_2 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_172, [0, 2, 3]), kwargs = {})
#   %convert_element_type_158 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_52, torch.float32), kwargs = {})
#   %sub_53 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_158, %unsqueeze_214), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_172, %sub_53), kwargs = {})
#   %sum_3 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_371, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_0 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 259006464, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + 2048*(r0_2 // 49) + 16384*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr2 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = 0.02040816326530612
        tmp3 = tmp1 * tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp6 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask, tmp17, _tmp16)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5s/c5sc6khgnf2h6427gdkd65o7opuj2ybzbw7q6bb5pxaqeyj3hj24.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.div, aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %div), kwargs = {})
#   %convert_element_type_172 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %sum_2 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_172, [0, 2, 3]), kwargs = {})
triton_per_fused_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_per_fused_div_native_batch_norm_backward_threshold_backward_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 540672, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_div_native_batch_norm_backward_threshold_backward_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hw/chwtualnalwb45xnk6elrxkyodhhmawwy43a7exvphsbrrnpt6ry.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_157], Original ATen: [aten.div, aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_157 => convert_element_type_158
#   scalar_tensor => full_default
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %div), kwargs = {})
#   %convert_element_type_172 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %convert_element_type_158 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_52, torch.float32), kwargs = {})
#   %sub_53 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_158, %unsqueeze_214), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_172, %sub_53), kwargs = {})
#   %sum_3 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_371, [0, 2, 3]), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_3, %squeeze_157), kwargs = {})
triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 565248, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ga/cga7wnutgxs6crr5kjvkvjeyst5okeu5gxnz4ckugwp3xuezm5dj.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_157], Original ATen: [aten.div, aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_157 => convert_element_type_158
#   scalar_tensor => full_default
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %div), kwargs = {})
#   %convert_element_type_172 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %convert_element_type_158 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_52, torch.float32), kwargs = {})
#   %sub_53 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_158, %unsqueeze_214), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_220), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_172, %mul_377), kwargs = {})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_55, %unsqueeze_217), kwargs = {})
#   %mul_378 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %unsqueeze_223), kwargs = {})
#   %convert_element_type_174 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_378, torch.bfloat16), kwargs = {})
#   %convolution_backward : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_174, %relu_47, %convert_element_type_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i1', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 361799680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 2048)
    x2 = xindex // 100352
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + 2048*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.02040816326530612
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = 3.985969387755102e-05
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp6 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jh/cjhcjfjftwhdkfltzxotfdl26svwcdoqguevr5va7tmnzy27qhkr.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_154], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_154 => convert_element_type_155
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_47, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default, %getitem_108), kwargs = {})
#   %convert_element_type_176 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_1, torch.float32), kwargs = {})
#   %sum_4 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_176, [0, 2, 3]), kwargs = {})
#   %convert_element_type_155 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_51, torch.float32), kwargs = {})
#   %sub_57 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_155, %unsqueeze_226), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_176, %sub_57), kwargs = {})
#   %sum_5 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_380, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 78678016, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 65536*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 65536*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 512*r0_2 + 65536*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/zs/czsuqdg3fr2qkpc274xe7qkq6zfozriyotltd32pqw6wueq3s47k.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_47, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default, %getitem_108), kwargs = {})
#   %convert_element_type_176 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_1, torch.float32), kwargs = {})
#   %sum_4 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_176, [0, 2, 3]), kwargs = {})
triton_red_fused_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 405504, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 196
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vi/cviw3wr3ixanh5epeppwxrxxgxfn7r7msppkc3f465tstvshpefr.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_154], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_154 => convert_element_type_155
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_47, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default, %getitem_108), kwargs = {})
#   %convert_element_type_176 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_1, torch.float32), kwargs = {})
#   %convert_element_type_155 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_51, torch.float32), kwargs = {})
#   %sub_57 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_155, %unsqueeze_226), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_176, %sub_57), kwargs = {})
#   %sum_5 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_380, [0, 2, 3]), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_5, %squeeze_154), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 411648, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 196
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/hu/chuardif5ucq5yx4i34mtijimrxtktxrf6xjiywwuvvvreee4roi.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_154], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_154 => convert_element_type_155
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_47, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default, %getitem_108), kwargs = {})
#   %convert_element_type_176 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_1, torch.float32), kwargs = {})
#   %convert_element_type_155 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_51, torch.float32), kwargs = {})
#   %sub_57 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_155, %unsqueeze_226), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_232), kwargs = {})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_176, %mul_386), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_59, %unsqueeze_229), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_235), kwargs = {})
#   %convert_element_type_178 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_387, torch.bfloat16), kwargs = {})
#   %convolution_backward_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_178, %relu_46, %convert_element_type_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 128460800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 3.985969387755102e-05
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/kr/ckrfmgdqgpo4cpygg7f65gu4vn2ihmfo54353q6xuzehp77bj35s.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_147], Original ATen: [aten.div, aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_147 => convert_element_type_149
#   scalar_tensor => full_default
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %div), kwargs = {})
#   %add_281 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %getitem_114), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_45, 0), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default, %add_281), kwargs = {})
#   %convert_element_type_184 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_3, torch.float32), kwargs = {})
#   %sum_8 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_184, [0, 2, 3]), kwargs = {})
#   %convert_element_type_149 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_49, torch.float32), kwargs = {})
#   %sub_65 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_149, %unsqueeze_250), kwargs = {})
#   %mul_398 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_184, %sub_65), kwargs = {})
#   %sum_9 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_398, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_add_div_native_batch_norm_backward_threshold_backward_8 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_add_div_native_batch_norm_backward_threshold_backward_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i1', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_div_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 464527360, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_div_native_batch_norm_backward_threshold_backward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp4 = tl.load(in_ptr2 + (x0 + 2048*(r0_2 // 49) + 16384*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr4 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = 0.02040816326530612
        tmp6 = tmp4 * tmp5
        tmp7 = tl.where(tmp3, tmp1, tmp6)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.where(tmp2, tmp1, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp11 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask, tmp22, _tmp21)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/dz/cdzg732ifxmb2oov3i65co2x6pt4yf72x4c3fd6zj4ld3ubph3wf.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_147], Original ATen: [aten.div, aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_147 => convert_element_type_149
#   scalar_tensor => full_default
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %div), kwargs = {})
#   %add_281 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %getitem_114), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_45, 0), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default, %add_281), kwargs = {})
#   %convert_element_type_184 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_3, torch.float32), kwargs = {})
#   %convert_element_type_149 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_49, torch.float32), kwargs = {})
#   %sub_65 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_149, %unsqueeze_250), kwargs = {})
#   %mul_404 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_256), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_184, %mul_404), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_67, %unsqueeze_253), kwargs = {})
#   %mul_405 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_259), kwargs = {})
#   %convert_element_type_186 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_405, torch.bfloat16), kwargs = {})
#   %convolution_backward_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_186, %relu_44, %convert_element_type_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i1', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 567320576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = (xindex % 2048)
    x2 = xindex // 100352
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x4), None).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x0 + 2048*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x4), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x4), None).to(tl.float32)
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = 0.02040816326530612
    tmp6 = tmp4 * tmp5
    tmp7 = tl.where(tmp3, tmp1, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp13 - tmp14
    tmp17 = 3.985969387755102e-05
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp11 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.store(out_ptr1 + (x4), tmp30, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/bo/cbogspqchwykcwhlwc7vbw75fx5ylbhfhdmp4tdutmjzdub5hezh.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.div, aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 49), kwargs = {})
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %div), kwargs = {})
#   %add_281 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %getitem_114), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_45, 0), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default, %add_281), kwargs = {})
#   %add_282 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_3, %getitem_123), kwargs = {})
#   %le_6 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_42, 0), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %full_default, %add_282), kwargs = {})
#   %convert_element_type_196 : [num_users=6] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_6, torch.float32), kwargs = {})
triton_poi_fused_add_div_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_poi_fused_add_div_native_batch_norm_backward_threshold_backward_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*i1', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 875560960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_native_batch_norm_backward_threshold_backward_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 2048)
    x2 = xindex // 100352
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x3), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), None).to(tl.int1)
    tmp6 = tl.load(in_ptr3 + (x0 + 2048*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (x3), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = 0.02040816326530612
    tmp8 = tmp6 * tmp7
    tmp9 = tl.where(tmp5, tmp1, tmp8)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.where(tmp4, tmp1, tmp11)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/dc/cdcgil7vajwopbdfwkgve252gl4vhx2upvqcfaqdwznaxk5atahb.py
# Topologically Sorted Source Nodes: [input_8, out_137], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_8 => convert_element_type_140
#   out_137 => convert_element_type_137
# Graph fragment:
#   %sum_14 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_196, [0, 2, 3]), kwargs = {})
#   %convert_element_type_140 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_46, torch.float32), kwargs = {})
#   %sub_77 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_140, %unsqueeze_286), kwargs = {})
#   %mul_425 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_196, %sub_77), kwargs = {})
#   %sum_15 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_425, [0, 2, 3]), kwargs = {})
#   %sum_16 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_196, [0, 2, 3]), kwargs = {})
#   %convert_element_type_137 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_45, torch.float32), kwargs = {})
#   %sub_81 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_137, %unsqueeze_298), kwargs = {})
#   %mul_434 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_196, %sub_81), kwargs = {})
#   %sum_17 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_434, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_11 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 415252480, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (x0 + 2048*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp0 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(out_ptr2 + (x3), tmp2, None)
    tl.store(out_ptr3 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/c2/cc2augrtcaliafq7nhk4gnqh5hrdeinp5wj5uove2zdizrkwt22d.py
# Topologically Sorted Source Nodes: [input_8, out_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_8 => convert_element_type_140
#   out_137 => convert_element_type_137
# Graph fragment:
#   %convert_element_type_140 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_46, torch.float32), kwargs = {})
#   %sub_77 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_140, %unsqueeze_286), kwargs = {})
#   %mul_431 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_292), kwargs = {})
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_196, %mul_431), kwargs = {})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_79, %unsqueeze_289), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_295), kwargs = {})
#   %convert_element_type_198 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_432, torch.bfloat16), kwargs = {})
#   %convolution_backward_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_198, %relu_39, %convert_element_type_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   %convert_element_type_137 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_45, torch.float32), kwargs = {})
#   %sub_81 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_137, %unsqueeze_298), kwargs = {})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_304), kwargs = {})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_196, %mul_440), kwargs = {})
#   %sub_84 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_83, %unsqueeze_301), kwargs = {})
#   %mul_441 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_84, %unsqueeze_307), kwargs = {})
#   %convert_element_type_202 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_441, torch.bfloat16), kwargs = {})
#   %convolution_backward_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_202, %relu_41, %convert_element_type_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822165504}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), None).to(tl.float32)
    tmp22 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 3.985969387755102e-05
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 - tmp22
    tmp25 = tmp24 * tmp6
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp0 - tmp29
    tmp32 = tmp31 * tmp6
    tmp33 = tmp30 - tmp32
    tmp35 = tmp26 * tmp34
    tmp36 = tmp33 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
    tl.store(in_out_ptr1 + (x2), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gc/cgchwqz2uzlqo5ljj4uaoogpn2dnylgykgt7hby2nfgqohy27lcd.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_131], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_131 => convert_element_type_131
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_8 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_40, 0), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default, %getitem_132), kwargs = {})
#   %convert_element_type_208 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_8, torch.float32), kwargs = {})
#   %sum_20 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_208, [0, 2, 3]), kwargs = {})
#   %convert_element_type_131 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_43, torch.float32), kwargs = {})
#   %sub_89 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_131, %unsqueeze_322), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_208, %sub_89), kwargs = {})
#   %sum_21 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_452, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_13 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 310380544, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 200704*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 200704*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 512*r0_2 + 200704*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jp/cjp7ftvyckbzqyrbo24h4prt4gfhtyylbrybgbvzii3abmk7rqsn.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_8 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_40, 0), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default, %getitem_132), kwargs = {})
#   %convert_element_type_208 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_8, torch.float32), kwargs = {})
#   %sum_20 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_208, [0, 2, 3]), kwargs = {})
triton_red_fused_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 528384, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_14(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/iv/civglgntuuc7x4xifhapulphamwsoy3qifbwcdvlznndhfmcozri.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_131], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_131 => convert_element_type_131
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_8 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_40, 0), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default, %getitem_132), kwargs = {})
#   %convert_element_type_208 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_8, torch.float32), kwargs = {})
#   %convert_element_type_131 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_43, torch.float32), kwargs = {})
#   %sub_89 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_131, %unsqueeze_322), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_208, %sub_89), kwargs = {})
#   %sum_21 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_452, [0, 2, 3]), kwargs = {})
#   %mul_460 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_21, %squeeze_130), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_15 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 534528, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_15(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xm/cxm7lenqdrqbc2h3vb74lmhdewpzh2fciepvmxaiqe6r2gqguvrd.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_131], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_131 => convert_element_type_131
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_8 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_40, 0), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default, %getitem_132), kwargs = {})
#   %convert_element_type_208 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_8, torch.float32), kwargs = {})
#   %convert_element_type_131 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_43, torch.float32), kwargs = {})
#   %sub_89 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_131, %unsqueeze_322), kwargs = {})
#   %mul_458 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_328), kwargs = {})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_208, %mul_458), kwargs = {})
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_91, %unsqueeze_325), kwargs = {})
#   %mul_459 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_331), kwargs = {})
#   %convert_element_type_210 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_459, torch.bfloat16), kwargs = {})
#   %convolution_backward_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_210, %relu_39, %convert_element_type_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 513812480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 9.964923469387754e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5x/c5x3r4twwgyqdsy3duqnndbala2pdoucqvdsabdats3rqezxhbmm.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_127], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_127 => convert_element_type_128
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_283 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_126, %getitem_135), kwargs = {})
#   %le_9 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_39, 0), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %add_283), kwargs = {})
#   %convert_element_type_212 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_9, torch.float32), kwargs = {})
#   %sum_22 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_212, [0, 2, 3]), kwargs = {})
#   %convert_element_type_128 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_42, torch.float32), kwargs = {})
#   %sub_93 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_128, %unsqueeze_334), kwargs = {})
#   %mul_461 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_212, %sub_93), kwargs = {})
#   %sum_23 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_461, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 824184832, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pe/cpefnyobz3njvv5zdtbusz37wudvjip6bbxj4uup47mkmkndrl3b.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_283 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_126, %getitem_135), kwargs = {})
#   %le_9 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_39, 0), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %add_283), kwargs = {})
#   %convert_element_type_212 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_9, torch.float32), kwargs = {})
#   %sum_22 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_212, [0, 2, 3]), kwargs = {})
triton_red_fused_add_native_batch_norm_backward_threshold_backward_18 = async_compile.triton('triton_red_fused_add_native_batch_norm_backward_threshold_backward_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 532480, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_add_native_batch_norm_backward_threshold_backward_18(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/62/c62qs7j4cu7pphgos47hprrxlx3kdq2ujxcbq4xfmsmixqew372a.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_127], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_127 => convert_element_type_128
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_283 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_126, %getitem_135), kwargs = {})
#   %le_9 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_39, 0), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %add_283), kwargs = {})
#   %convert_element_type_212 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_9, torch.float32), kwargs = {})
#   %convert_element_type_128 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_42, torch.float32), kwargs = {})
#   %sub_93 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_128, %unsqueeze_334), kwargs = {})
#   %mul_461 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_212, %sub_93), kwargs = {})
#   %sum_23 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_461, [0, 2, 3]), kwargs = {})
#   %mul_469 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_23, %squeeze_127), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 544768, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/6r/c6r6gzjftp2wzfay4jotlzm5fnqb3wumocsbtb6dnoce2iqti637.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_127], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_127 => convert_element_type_128
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_283 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_126, %getitem_135), kwargs = {})
#   %le_9 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_39, 0), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %add_283), kwargs = {})
#   %convert_element_type_212 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_9, torch.float32), kwargs = {})
#   %convert_element_type_128 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_42, torch.float32), kwargs = {})
#   %sub_93 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_128, %unsqueeze_334), kwargs = {})
#   %mul_467 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_340), kwargs = {})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_212, %mul_467), kwargs = {})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_95, %unsqueeze_337), kwargs = {})
#   %mul_468 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_343), kwargs = {})
#   %convert_element_type_214 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_468, torch.bfloat16), kwargs = {})
#   %convolution_backward_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_214, %relu_38, %convert_element_type_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1233145856}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 9.964923469387754e-06
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/px/cpx25uxeuq44b6ndswcqvkjdkmvqa4nehmjmhsxdban37vtux6g3.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_124], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_124 => convert_element_type_125
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_10 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_38, 0), kwargs = {})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_10, %full_default, %getitem_138), kwargs = {})
#   %convert_element_type_216 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_10, torch.float32), kwargs = {})
#   %sum_24 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_216, [0, 2, 3]), kwargs = {})
#   %convert_element_type_125 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_41, torch.float32), kwargs = {})
#   %sub_97 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_125, %unsqueeze_346), kwargs = {})
#   %mul_470 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_216, %sub_97), kwargs = {})
#   %sum_25 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_470, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 156238848, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 50176*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 50176*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 256*r0_2 + 50176*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ih/cihkyetgji7ytx6t2iif6rcwae6jrqgbnbis3svmuga5qtgirp6j.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_10 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_38, 0), kwargs = {})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_10, %full_default, %getitem_138), kwargs = {})
#   %convert_element_type_216 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_10, torch.float32), kwargs = {})
#   %sum_24 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_216, [0, 2, 3]), kwargs = {})
triton_red_fused_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 526336, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_22(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/6v/c6v5rlzh3snheassmsrqxwqfbfnkgnzpszzedbzf3nvox324tmy7.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_124], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_124 => convert_element_type_125
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_10 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_38, 0), kwargs = {})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_10, %full_default, %getitem_138), kwargs = {})
#   %convert_element_type_216 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_10, torch.float32), kwargs = {})
#   %convert_element_type_125 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_41, torch.float32), kwargs = {})
#   %sub_97 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_125, %unsqueeze_346), kwargs = {})
#   %mul_470 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_216, %sub_97), kwargs = {})
#   %sum_25 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_470, [0, 2, 3]), kwargs = {})
#   %mul_478 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_25, %squeeze_124), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 529408, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/p5/cp5l2bblsv4mmnz4y7malhhmqpg32dvnpicckkiv57tahw4ujqzm.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_124], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_124 => convert_element_type_125
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_10 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_38, 0), kwargs = {})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_10, %full_default, %getitem_138), kwargs = {})
#   %convert_element_type_216 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_10, torch.float32), kwargs = {})
#   %convert_element_type_125 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_41, torch.float32), kwargs = {})
#   %sub_97 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_125, %unsqueeze_346), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_97, %unsqueeze_352), kwargs = {})
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_216, %mul_476), kwargs = {})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_99, %unsqueeze_349), kwargs = {})
#   %mul_477 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_355), kwargs = {})
#   %convert_element_type_218 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_477, torch.bfloat16), kwargs = {})
#   %convolution_backward_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_218, %relu_37, %convert_element_type_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256906240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 9.964923469387754e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/so/csongjy6uujuhbz33i7774n5mhwwu5gfb3din7fswu7ru47brjkz.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_283 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_126, %getitem_135), kwargs = {})
#   %le_9 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_39, 0), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default, %add_283), kwargs = {})
#   %add_284 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_9, %getitem_144), kwargs = {})
#   %le_12 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_36, 0), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_12, %full_default, %add_284), kwargs = {})
triton_poi_fused_add_threshold_backward_25 = async_compile.triton('triton_poi_fused_add_threshold_backward_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1438646272}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_threshold_backward_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gs/cgsilzc5utnwms5bqlfdxuu4pzpmf3mcjmjqcucbflxworh5wgnb.py
# Topologically Sorted Source Nodes: [out_117], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_117 => convert_element_type_119
# Graph fragment:
#   %convert_element_type_224 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_12, torch.float32), kwargs = {})
#   %sum_28 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_224, [0, 2, 3]), kwargs = {})
#   %convert_element_type_119 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_39, torch.float32), kwargs = {})
#   %sub_105 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_119, %unsqueeze_370), kwargs = {})
#   %mul_488 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_224, %sub_105), kwargs = {})
#   %sum_29 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_488, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_26 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 413143040, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_26(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp1 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/aj/cajbao6xb7jsigkee4npxvmogfb3axgoevxucxbaujetrwdiabvy.py
# Topologically Sorted Source Nodes: [out_117], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_117 => convert_element_type_119
# Graph fragment:
#   %convert_element_type_224 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_12, torch.float32), kwargs = {})
#   %convert_element_type_119 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_39, torch.float32), kwargs = {})
#   %sub_105 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_119, %unsqueeze_370), kwargs = {})
#   %mul_494 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_376), kwargs = {})
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_224, %mul_494), kwargs = {})
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_107, %unsqueeze_373), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %unsqueeze_379), kwargs = {})
#   %convert_element_type_226 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_495, torch.bfloat16), kwargs = {})
#   %convolution_backward_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_226, %relu_35, %convert_element_type_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822104064}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 9.964923469387754e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp1 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jj/cjj2gynup5dtbuzb4xf3pcpxvpjmaqdviow4rnpnz45heswn4lld.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_287 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_18, %getitem_171), kwargs = {})
#   %le_21 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_27, 0), kwargs = {})
#   %where_21 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_21, %full_default, %add_287), kwargs = {})
#   %add_288 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_21, %getitem_180), kwargs = {})
#   %le_24 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_24, 0), kwargs = {})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_24, %full_default, %add_288), kwargs = {})
#   %convert_element_type_272 : [num_users=6] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_24, torch.float32), kwargs = {})
triton_poi_fused_add_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_poi_fused_add_native_batch_norm_backward_threshold_backward_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1849688064}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_batch_norm_backward_threshold_backward_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5b/c5b7mgdnsxdydtov2kjnpp2ki3y5v6itr4wqmnaq3oppmipgrttt.py
# Topologically Sorted Source Nodes: [input_6, out_77], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_6 => convert_element_type_83
#   out_77 => convert_element_type_80
# Graph fragment:
#   %sum_52 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_272, [0, 2, 3]), kwargs = {})
#   %convert_element_type_83 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_27, torch.float32), kwargs = {})
#   %sub_153 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_83, %unsqueeze_514), kwargs = {})
#   %mul_596 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_272, %sub_153), kwargs = {})
#   %sum_53 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_596, [0, 2, 3]), kwargs = {})
#   %sum_54 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_272, [0, 2, 3]), kwargs = {})
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_26, torch.float32), kwargs = {})
#   %sub_157 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_80, %unsqueeze_526), kwargs = {})
#   %mul_605 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_272, %sub_157), kwargs = {})
#   %sum_55 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_605, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_29 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 826286080, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (x0 + 1024*r0_2 + 802816*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp0 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp10, None)
    tl.store(out_ptr2 + (x3), tmp2, None)
    tl.store(out_ptr3 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/uv/cuvovfq4n2y3ibroutm6pnylfjinef4ioweyqc2btkvhxdbxalub.py
# Topologically Sorted Source Nodes: [input_6, out_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_6 => convert_element_type_83
#   out_77 => convert_element_type_80
# Graph fragment:
#   %convert_element_type_83 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_27, torch.float32), kwargs = {})
#   %sub_153 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_83, %unsqueeze_514), kwargs = {})
#   %mul_602 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_153, %unsqueeze_520), kwargs = {})
#   %sub_155 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_272, %mul_602), kwargs = {})
#   %sub_156 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_155, %unsqueeze_517), kwargs = {})
#   %mul_603 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_156, %unsqueeze_523), kwargs = {})
#   %convert_element_type_274 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_603, torch.bfloat16), kwargs = {})
#   %convolution_backward_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_274, %relu_21, %convert_element_type_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_26, torch.float32), kwargs = {})
#   %sub_157 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_80, %unsqueeze_526), kwargs = {})
#   %mul_611 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_157, %unsqueeze_532), kwargs = {})
#   %sub_159 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_272, %mul_611), kwargs = {})
#   %sub_160 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_159, %unsqueeze_529), kwargs = {})
#   %mul_612 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_160, %unsqueeze_535), kwargs = {})
#   %convert_element_type_278 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_612, torch.bfloat16), kwargs = {})
#   %convolution_backward_26 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_278, %relu_23, %convert_element_type_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1644208128}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_30(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), None).to(tl.float32)
    tmp22 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 9.964923469387754e-06
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 - tmp22
    tmp25 = tmp24 * tmp6
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp0 - tmp29
    tmp32 = tmp31 * tmp6
    tmp33 = tmp30 - tmp32
    tmp35 = tmp26 * tmp34
    tmp36 = tmp33 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
    tl.store(in_out_ptr1 + (x2), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/z4/cz4kye4wi5odharsbgsj2pjybzet24ssvoyjv46kddoceuqu7jmp.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_71], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_71 => convert_element_type_74
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_26 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_22, 0), kwargs = {})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_26, %full_default, %getitem_189), kwargs = {})
#   %convert_element_type_284 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_26, torch.float32), kwargs = {})
#   %sum_58 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_284, [0, 2, 3]), kwargs = {})
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_24, torch.float32), kwargs = {})
#   %sub_165 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_74, %unsqueeze_550), kwargs = {})
#   %mul_623 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_284, %sub_165), kwargs = {})
#   %sum_59 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_623, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 618660864, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 200704*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 200704*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 256*r0_2 + 200704*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/p6/cp63ph245ewxqpyazt5zygfesdrwtgfdiynqd4bn5tpi5dm7fols.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_71], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_71 => convert_element_type_74
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_26 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_22, 0), kwargs = {})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_26, %full_default, %getitem_189), kwargs = {})
#   %convert_element_type_284 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_26, torch.float32), kwargs = {})
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_24, torch.float32), kwargs = {})
#   %sub_165 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_74, %unsqueeze_550), kwargs = {})
#   %mul_629 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_165, %unsqueeze_556), kwargs = {})
#   %sub_167 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_284, %mul_629), kwargs = {})
#   %sub_168 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_167, %unsqueeze_553), kwargs = {})
#   %mul_630 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_168, %unsqueeze_559), kwargs = {})
#   %convert_element_type_286 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_630, torch.bfloat16), kwargs = {})
#   %convolution_backward_28 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_286, %relu_21, %convert_element_type_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1027609600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 2.4912308673469386e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/la/cla3igfdqtvg2js7wac2jlkk7uoq5dbtgijkt4deklhogn5eg6ic.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_67], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_67 => convert_element_type_71
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_289 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_183, %getitem_192), kwargs = {})
#   %le_27 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_21, 0), kwargs = {})
#   %where_27 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_27, %full_default, %add_289), kwargs = {})
#   %convert_element_type_288 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_27, torch.float32), kwargs = {})
#   %sum_60 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_288, [0, 2, 3]), kwargs = {})
#   %convert_element_type_71 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_23, torch.float32), kwargs = {})
#   %sub_169 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_71, %unsqueeze_562), kwargs = {})
#   %mul_632 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_288, %sub_169), kwargs = {})
#   %sum_61 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_632, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_33 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1645774848, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lz/clzzrtkaup6bq2zza6qfsuen6igdlmwlcir3c546pxw4ftsiyi5t.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_67], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_67 => convert_element_type_71
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_289 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_183, %getitem_192), kwargs = {})
#   %le_27 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_21, 0), kwargs = {})
#   %where_27 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_27, %full_default, %add_289), kwargs = {})
#   %convert_element_type_288 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_27, torch.float32), kwargs = {})
#   %convert_element_type_71 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_23, torch.float32), kwargs = {})
#   %sub_169 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_71, %unsqueeze_562), kwargs = {})
#   %mul_638 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_169, %unsqueeze_568), kwargs = {})
#   %sub_171 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_288, %mul_638), kwargs = {})
#   %sub_172 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_171, %unsqueeze_565), kwargs = {})
#   %mul_639 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_172, %unsqueeze_571), kwargs = {})
#   %convert_element_type_290 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_639, torch.bfloat16), kwargs = {})
#   %convolution_backward_29 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_290, %relu_20, %convert_element_type_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2466260992}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 2.4912308673469386e-06
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gs/cgsontnitopwtgizwtekxmcc2psx6yg5xx7mjdbudfbdjgppodx2.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_64], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_64 => convert_element_type_68
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_28 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_20, 0), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_28, %full_default, %getitem_195), kwargs = {})
#   %convert_element_type_292 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_28, torch.float32), kwargs = {})
#   %sum_62 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_292, [0, 2, 3]), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_22, torch.float32), kwargs = {})
#   %sub_173 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_68, %unsqueeze_574), kwargs = {})
#   %mul_641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_292, %sub_173), kwargs = {})
#   %sum_63 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_641, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 310379008, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 50176*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 50176*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 50176*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yt/cytgvwbx4nrxcyfu3yr65ju7f2227tcozmeb2ik7rvmx6z3nuj56.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_28 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_20, 0), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_28, %full_default, %getitem_195), kwargs = {})
#   %convert_element_type_292 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_28, torch.float32), kwargs = {})
#   %sum_62 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_292, [0, 2, 3]), kwargs = {})
triton_red_fused_native_batch_norm_backward_threshold_backward_36 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 525312, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_36(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pc/cpcdeyjo7jgtbrpaljbovx7a2v3eth6lt7kczfewnrufasq5t2je.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_64], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_64 => convert_element_type_68
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_28 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_20, 0), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_28, %full_default, %getitem_195), kwargs = {})
#   %convert_element_type_292 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_28, torch.float32), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_22, torch.float32), kwargs = {})
#   %sub_173 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_68, %unsqueeze_574), kwargs = {})
#   %mul_641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_292, %sub_173), kwargs = {})
#   %sum_63 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_641, [0, 2, 3]), kwargs = {})
#   %mul_649 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_63, %squeeze_67), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 526848, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/cz/cczysuzmuakdospxgqtc4s5ixl2bvr6jxqra6napymf4vpczf7re.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_64], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_64 => convert_element_type_68
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_28 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_20, 0), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_28, %full_default, %getitem_195), kwargs = {})
#   %convert_element_type_292 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_28, torch.float32), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_22, torch.float32), kwargs = {})
#   %sub_173 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_68, %unsqueeze_574), kwargs = {})
#   %mul_647 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_173, %unsqueeze_580), kwargs = {})
#   %sub_175 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_292, %mul_647), kwargs = {})
#   %sub_176 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_175, %unsqueeze_577), kwargs = {})
#   %mul_648 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_176, %unsqueeze_583), kwargs = {})
#   %convert_element_type_294 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_648, torch.bfloat16), kwargs = {})
#   %convolution_backward_30 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_294, %relu_19, %convert_element_type_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 513804800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 2.4912308673469386e-06
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/da/cdaaw4c2v5yfzy2d5kc7udu7kgaerm7z5f4wrfupvlyzvh6evuhw.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_289 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_183, %getitem_192), kwargs = {})
#   %le_27 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_21, 0), kwargs = {})
#   %where_27 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_27, %full_default, %add_289), kwargs = {})
#   %add_290 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_27, %getitem_201), kwargs = {})
#   %le_30 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_18, 0), kwargs = {})
#   %where_30 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_30, %full_default, %add_290), kwargs = {})
triton_poi_fused_add_threshold_backward_39 = async_compile.triton('triton_poi_fused_add_threshold_backward_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2877292544}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_threshold_backward_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vp/cvpyjdffrcgvwgu3ukxobdfqtqvdfcmwx7xaeawdxnxw3nmlihe5.py
# Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_57 => convert_element_type_62
# Graph fragment:
#   %convert_element_type_300 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_30, torch.float32), kwargs = {})
#   %sum_66 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_300, [0, 2, 3]), kwargs = {})
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_20, torch.float32), kwargs = {})
#   %sub_181 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_62, %unsqueeze_598), kwargs = {})
#   %mul_659 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_300, %sub_181), kwargs = {})
#   %sum_67 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_659, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 823691264, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp1 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lk/clklzcbxwx6fanvpkxi55eroul46hu5shf7xtcvbik7fb5vnhxvf.py
# Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_57 => convert_element_type_62
# Graph fragment:
#   %convert_element_type_300 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_30, torch.float32), kwargs = {})
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_20, torch.float32), kwargs = {})
#   %sub_181 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_62, %unsqueeze_598), kwargs = {})
#   %mul_665 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_181, %unsqueeze_604), kwargs = {})
#   %sub_183 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_300, %mul_665), kwargs = {})
#   %sub_184 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_183, %unsqueeze_601), kwargs = {})
#   %mul_666 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_184, %unsqueeze_607), kwargs = {})
#   %convert_element_type_302 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_666, torch.bfloat16), kwargs = {})
#   %convolution_backward_32 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_302, %relu_17, %convert_element_type_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1644177408}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 2.4912308673469386e-06
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp1 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/6q/c6qxijywtphar23bxdl4dilhlxbhjw66u3kltxnvoem3pthztxd3.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_291 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_30, %getitem_210), kwargs = {})
#   %le_33 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_15, 0), kwargs = {})
#   %where_33 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_33, %full_default, %add_291), kwargs = {})
#   %add_292 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_33, %getitem_219), kwargs = {})
#   %le_36 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_12, 0), kwargs = {})
#   %where_36 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_36, %full_default, %add_292), kwargs = {})
#   %convert_element_type_324 : [num_users=6] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_36, torch.float32), kwargs = {})
triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3699376128}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/im/cimpzsorglolaz2cwpph7fra2ie325tfmyct2gbynqngzw4inby5.py
# Topologically Sorted Source Nodes: [input_4, out_37], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_4 => convert_element_type_44
#   out_37 => convert_element_type_41
# Graph fragment:
#   %sum_78 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_324, [0, 2, 3]), kwargs = {})
#   %convert_element_type_44 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_14, torch.float32), kwargs = {})
#   %sub_205 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_44, %unsqueeze_670), kwargs = {})
#   %mul_713 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_324, %sub_205), kwargs = {})
#   %sum_79 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_713, [0, 2, 3]), kwargs = {})
#   %sum_80 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_324, [0, 2, 3]), kwargs = {})
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_13, torch.float32), kwargs = {})
#   %sub_209 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_41, %unsqueeze_682), kwargs = {})
#   %mul_722 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_324, %sub_209), kwargs = {})
#   %sum_81 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_722, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_43 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1647382528, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (x0 + 512*r0_2 + 1048576*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp0 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(out_ptr2 + (x3), tmp2, xmask)
    tl.store(out_ptr3 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/zr/czr7htu7qihxtd575sfhpf7yu4kz6a6crxke6biz5spuoqzhvmin.py
# Topologically Sorted Source Nodes: [input_4, out_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_4 => convert_element_type_44
#   out_37 => convert_element_type_41
# Graph fragment:
#   %convert_element_type_44 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_14, torch.float32), kwargs = {})
#   %sub_205 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_44, %unsqueeze_670), kwargs = {})
#   %mul_719 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_205, %unsqueeze_676), kwargs = {})
#   %sub_207 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_324, %mul_719), kwargs = {})
#   %sub_208 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_207, %unsqueeze_673), kwargs = {})
#   %mul_720 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_208, %unsqueeze_679), kwargs = {})
#   %convert_element_type_326 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_720, torch.bfloat16), kwargs = {})
#   %convolution_backward_38 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_326, %relu_9, %convert_element_type_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_13, torch.float32), kwargs = {})
#   %sub_209 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_41, %unsqueeze_682), kwargs = {})
#   %mul_728 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_209, %unsqueeze_688), kwargs = {})
#   %sub_211 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_324, %mul_728), kwargs = {})
#   %sub_212 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_211, %unsqueeze_685), kwargs = {})
#   %mul_729 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_212, %unsqueeze_691), kwargs = {})
#   %convert_element_type_330 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_729, torch.bfloat16), kwargs = {})
#   %convolution_backward_39 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_330, %relu_11, %convert_element_type_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_44', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3288354816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_44(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr1 + (x2), None).to(tl.float32)
    tmp22 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = 2.4912308673469386e-06
    tmp7 = tmp5 * tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 * tmp9
    tmp11 = tmp4 * tmp10
    tmp12 = tmp0 - tmp11
    tmp14 = tmp13 * tmp6
    tmp15 = tmp12 - tmp14
    tmp17 = tmp8 * tmp16
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 - tmp22
    tmp25 = tmp24 * tmp6
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp0 - tmp29
    tmp32 = tmp31 * tmp6
    tmp33 = tmp30 - tmp32
    tmp35 = tmp26 * tmp34
    tmp36 = tmp33 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
    tl.store(in_out_ptr1 + (x2), tmp37, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/er/ceri3clbyuftwmmv2xrg5ohy3wtoat5mzexch4tfyhctu32ows75.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_31 => convert_element_type_35
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_38 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_10, 0), kwargs = {})
#   %where_38 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_38, %full_default, %getitem_228), kwargs = {})
#   %convert_element_type_336 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_38, torch.float32), kwargs = {})
#   %sum_84 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_336, [0, 2, 3]), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %sub_217 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_35, %unsqueeze_706), kwargs = {})
#   %mul_740 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_336, %sub_217), kwargs = {})
#   %sum_85 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_740, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1234731520, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 262144*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 262144*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 262144*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gf/cgfjnwyagt5kykyha2yqyjpuzumbdjbcy7n2fdpf5aanab2etu6d.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_38 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_10, 0), kwargs = {})
#   %where_38 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_38, %full_default, %getitem_228), kwargs = {})
#   %convert_element_type_336 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_38, torch.float32), kwargs = {})
#   %sum_84 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_336, [0, 2, 3]), kwargs = {})
triton_red_fused_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 402432, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_46(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 784
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/om/com5rxwycd2sjni4qeyn7zaxzvm4gcolv32w26ukse7o56egu4iq.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_31 => convert_element_type_35
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_38 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_10, 0), kwargs = {})
#   %where_38 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_38, %full_default, %getitem_228), kwargs = {})
#   %convert_element_type_336 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_38, torch.float32), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %sub_217 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_35, %unsqueeze_706), kwargs = {})
#   %mul_740 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_336, %sub_217), kwargs = {})
#   %sum_85 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_740, [0, 2, 3]), kwargs = {})
#   %mul_748 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_85, %squeeze_34), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 403968, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 784
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jh/cjhgipqs6kezryoaj6zbu4ypk3crbv7736rmcpxa3rmlv4wc2ead.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_31 => convert_element_type_35
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_38 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_10, 0), kwargs = {})
#   %where_38 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_38, %full_default, %getitem_228), kwargs = {})
#   %convert_element_type_336 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_38, torch.float32), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_11, torch.float32), kwargs = {})
#   %sub_217 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_35, %unsqueeze_706), kwargs = {})
#   %mul_746 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_217, %unsqueeze_712), kwargs = {})
#   %sub_219 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_336, %mul_746), kwargs = {})
#   %sub_220 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_219, %unsqueeze_709), kwargs = {})
#   %mul_747 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_220, %unsqueeze_715), kwargs = {})
#   %convert_element_type_338 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_747, torch.bfloat16), kwargs = {})
#   %convolution_backward_41 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_338, %relu_9, %convert_element_type_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2055211520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205520896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 6.228077168367346e-07
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/oy/coyv4pbsot4dy2vum2hmf4zdjvy2zoipgptsjvtff2r2dl564azk.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_27], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_27 => convert_element_type_32
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_222, %getitem_231), kwargs = {})
#   %le_39 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
#   %where_39 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_39, %full_default, %add_293), kwargs = {})
#   %convert_element_type_340 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_39, torch.float32), kwargs = {})
#   %sum_86 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_340, [0, 2, 3]), kwargs = {})
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_10, torch.float32), kwargs = {})
#   %sub_221 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_32, %unsqueeze_718), kwargs = {})
#   %mul_749 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_340, %sub_221), kwargs = {})
#   %sum_87 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_749, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3291546624, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/c6/cc6bmwxxx5y4bqpsjenuov6adzfi7g2icavs3iy4skfql6c6kdp7.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_222, %getitem_231), kwargs = {})
#   %le_39 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
#   %where_39 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_39, %full_default, %add_293), kwargs = {})
#   %convert_element_type_340 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_39, torch.float32), kwargs = {})
#   %sum_86 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_340, [0, 2, 3]), kwargs = {})
triton_red_fused_add_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_red_fused_add_native_batch_norm_backward_threshold_backward_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 804864, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_add_native_batch_norm_backward_threshold_backward_50(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 784
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jv/cjvehjneyamrenuzph2t635oit4pdrsw46dk3dmsl5ei2pbmwj2t.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_27], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_27 => convert_element_type_32
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_222, %getitem_231), kwargs = {})
#   %le_39 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
#   %where_39 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_39, %full_default, %add_293), kwargs = {})
#   %convert_element_type_340 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_39, torch.float32), kwargs = {})
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_10, torch.float32), kwargs = {})
#   %sub_221 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_32, %unsqueeze_718), kwargs = {})
#   %mul_749 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_340, %sub_221), kwargs = {})
#   %sum_87 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_749, [0, 2, 3]), kwargs = {})
#   %mul_757 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_87, %squeeze_31), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 807936, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_51(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 784
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/io/ciof5gkm7j2udlk3cevrhgpyhyjuajqgkbwcrv4xpzwi73kxk2wx.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_27], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_27 => convert_element_type_32
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_222, %getitem_231), kwargs = {})
#   %le_39 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
#   %where_39 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_39, %full_default, %add_293), kwargs = {})
#   %convert_element_type_340 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_39, torch.float32), kwargs = {})
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_10, torch.float32), kwargs = {})
#   %sub_221 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_32, %unsqueeze_718), kwargs = {})
#   %mul_755 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_221, %unsqueeze_724), kwargs = {})
#   %sub_223 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_340, %mul_755), kwargs = {})
#   %sub_224 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_223, %unsqueeze_721), kwargs = {})
#   %mul_756 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_224, %unsqueeze_727), kwargs = {})
#   %convert_element_type_342 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_756, torch.bfloat16), kwargs = {})
#   %convolution_backward_42 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_342, %relu_8, %convert_element_type_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4932506624}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 6.228077168367346e-07
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/yn/cyni32oiwu6mtgvznw5nqehbex4rcno7qdmwlykolbdom3zd63og.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_24], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_24 => convert_element_type_29
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_40 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
#   %where_40 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_40, %full_default, %getitem_234), kwargs = {})
#   %convert_element_type_344 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_40, torch.float32), kwargs = {})
#   %sum_88 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_344, [0, 2, 3]), kwargs = {})
#   %convert_element_type_29 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_9, torch.float32), kwargs = {})
#   %sub_225 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_29, %unsqueeze_730), kwargs = {})
#   %mul_758 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_344, %sub_225), kwargs = {})
#   %sum_89 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_758, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 617611520, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 100352*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 64*r0_2 + 100352*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 64*r0_2 + 100352*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ih/cihj6l3rgi45dphubixrimfy2d6667u2f6cu4cpzyq2ra3fzdpn6.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_40 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
#   %where_40 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_40, %full_default, %getitem_234), kwargs = {})
#   %convert_element_type_344 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_40, torch.float32), kwargs = {})
#   %sum_88 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_344, [0, 2, 3]), kwargs = {})
triton_red_fused_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262656, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_54(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sg/csg74fj3qvu4y4dgrqdsz2yj77ayyc3inwtcwtrhbdwiiljhvnfd.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_24], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_24 => convert_element_type_29
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_40 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
#   %where_40 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_40, %full_default, %getitem_234), kwargs = {})
#   %convert_element_type_344 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_40, torch.float32), kwargs = {})
#   %convert_element_type_29 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_9, torch.float32), kwargs = {})
#   %sub_225 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_29, %unsqueeze_730), kwargs = {})
#   %mul_758 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_344, %sub_225), kwargs = {})
#   %sum_89 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_758, [0, 2, 3]), kwargs = {})
#   %mul_766 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_89, %squeeze_28), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 263424, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/72/c72fja424y6ciggypdqueqnaffmno2ok5euohturpluty3datd7p.py
# Topologically Sorted Source Nodes: [scalar_tensor, out_24], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_24 => convert_element_type_29
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_40 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_8, 0), kwargs = {})
#   %where_40 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_40, %full_default, %getitem_234), kwargs = {})
#   %convert_element_type_344 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_40, torch.float32), kwargs = {})
#   %convert_element_type_29 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_9, torch.float32), kwargs = {})
#   %sub_225 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_29, %unsqueeze_730), kwargs = {})
#   %mul_764 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_225, %unsqueeze_736), kwargs = {})
#   %sub_227 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_344, %mul_764), kwargs = {})
#   %sub_228 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_227, %unsqueeze_733), kwargs = {})
#   %mul_765 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_228, %unsqueeze_739), kwargs = {})
#   %convert_element_type_346 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_765, torch.bfloat16), kwargs = {})
#   %convolution_backward_43 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_346, %relu_7, %convert_element_type_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1027605760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 6.228077168367346e-07
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/xe/cxeasqoilgjeelsp62hn7n3wxq2yhfboidj2aumywm4sbxy6wpsu.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_222, %getitem_231), kwargs = {})
#   %le_39 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
#   %where_39 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_39, %full_default, %add_293), kwargs = {})
#   %add_294 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_39, %getitem_240), kwargs = {})
#   %le_42 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
#   %where_42 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_42, %full_default, %add_294), kwargs = {})
triton_poi_fused_add_threshold_backward_57 = async_compile.triton('triton_poi_fused_add_threshold_backward_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5754585088}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_threshold_backward_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 <= tmp1
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp4, tmp1, tmp7)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.where(tmp2, tmp1, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jt/cjt5wbgk6naxs2lezmhrmtt3lszpbyzd6wp4wttnkklufxlfp7fs.py
# Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   out_17 => convert_element_type_23
# Graph fragment:
#   %convert_element_type_352 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_42, torch.float32), kwargs = {})
#   %sum_92 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_352, [0, 2, 3]), kwargs = {})
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_7, torch.float32), kwargs = {})
#   %sub_233 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_23, %unsqueeze_754), kwargs = {})
#   %mul_776 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_352, %sub_233), kwargs = {})
#   %sum_93 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_776, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_58 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_58', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1647379456, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_58(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp1 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mv/cmv2a2awpaq75tlm4mo56pso4zyw2omzbzv7gmphie4lhb3jzx4q.py
# Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   out_17 => convert_element_type_23
# Graph fragment:
#   %convert_element_type_352 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_42, torch.float32), kwargs = {})
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_7, torch.float32), kwargs = {})
#   %sub_233 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_23, %unsqueeze_754), kwargs = {})
#   %mul_782 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_233, %unsqueeze_760), kwargs = {})
#   %sub_235 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_352, %mul_782), kwargs = {})
#   %sub_236 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_235, %unsqueeze_757), kwargs = {})
#   %mul_783 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_236, %unsqueeze_763), kwargs = {})
#   %convert_element_type_354 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_783, torch.bfloat16), kwargs = {})
#   %convolution_backward_45 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_354, %relu_5, %convert_element_type_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3288339456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 6.228077168367346e-07
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp1 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gt/cgtgyhbaet2kfy57av2n4zfqudgmyf44s66z76qwg5ozuauowrjn.py
# Topologically Sorted Source Nodes: [scalar_tensor, input_2, out_7], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_14
#   out_7 => convert_element_type_11
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_295 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_42, %getitem_249), kwargs = {})
#   %le_45 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
#   %where_45 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_45, %full_default, %add_295), kwargs = {})
#   %convert_element_type_364 : [num_users=6] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_45, torch.float32), kwargs = {})
#   %sum_98 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_364, [0, 2, 3]), kwargs = {})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_4, torch.float32), kwargs = {})
#   %sub_245 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_14, %unsqueeze_790), kwargs = {})
#   %mul_803 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_364, %sub_245), kwargs = {})
#   %sum_99 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_803, [0, 2, 3]), kwargs = {})
#   %sum_100 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_364, [0, 2, 3]), kwargs = {})
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %sub_249 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_11, %unsqueeze_802), kwargs = {})
#   %mul_812 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_364, %sub_249), kwargs = {})
#   %sum_101 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_812, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_60', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4116842496, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr5 + (x0 + 256*r0_2 + 524288*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp7 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask, tmp26, _tmp25)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
    tl.store(out_ptr2 + (x3), tmp9, None)
    tl.store(out_ptr3 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lj/cljs3duifim7atbcmpk3mmt7ej3k35baboitp3gl2rppz4yg4yke.py
# Topologically Sorted Source Nodes: [scalar_tensor, input_2, out_7], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_14
#   out_7 => convert_element_type_11
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_295 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where_42, %getitem_249), kwargs = {})
#   %le_45 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
#   %where_45 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_45, %full_default, %add_295), kwargs = {})
#   %convert_element_type_364 : [num_users=6] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_45, torch.float32), kwargs = {})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_4, torch.float32), kwargs = {})
#   %sub_245 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_14, %unsqueeze_790), kwargs = {})
#   %mul_809 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_245, %unsqueeze_796), kwargs = {})
#   %sub_247 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_364, %mul_809), kwargs = {})
#   %sub_248 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_247, %unsqueeze_793), kwargs = {})
#   %mul_810 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_248, %unsqueeze_799), kwargs = {})
#   %convert_element_type_366 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_810, torch.bfloat16), kwargs = {})
#   %convolution_backward_48 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_366, %getitem_2, %convert_element_type_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_3, torch.float32), kwargs = {})
#   %sub_249 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_11, %unsqueeze_802), kwargs = {})
#   %mul_818 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_249, %unsqueeze_808), kwargs = {})
#   %sub_251 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_364, %mul_818), kwargs = {})
#   %sub_252 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_251, %unsqueeze_805), kwargs = {})
#   %mul_819 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_252, %unsqueeze_811), kwargs = {})
#   %convert_element_type_370 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_819, torch.bfloat16), kwargs = {})
#   %convolution_backward_49 : [num_users=2] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_370, %relu_2, %convert_element_type_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_61 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*bf16', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7398762496}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x2), None).to(tl.float32)
    tmp28 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr13 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 6.228077168367346e-07
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp7 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp27 - tmp28
    tmp31 = tmp30 * tmp13
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp29 * tmp34
    tmp36 = tmp7 - tmp35
    tmp38 = tmp37 * tmp13
    tmp39 = tmp36 - tmp38
    tmp41 = tmp32 * tmp40
    tmp42 = tmp39 * tmp41
    tmp43 = tmp25.to(tl.float32)
    tmp44 = tmp42.to(tl.float32)
    tl.store(out_ptr2 + (x2), tmp43, None)
    tl.store(out_ptr3 + (x2), tmp44, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/4t/c4tn73tzypf5tnbluf53hacmcpmpnnfncaro4bbseexxjzhfxfr4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_296 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_252, %getitem_261), kwargs = {})
triton_poi_fused_add_62 = async_compile.triton('triton_poi_fused_add_62', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 822083584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_62(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/rh/crhn6g5xy2jpw4cpvq5lxavtutxae367uwq7y56rnuhibw5xmccx.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
# Source node to ATen node mapping:
#   x_3 => _low_memory_max_pool_offsets_to_indices
# Graph fragment:
#   %add_296 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_252, %getitem_261), kwargs = {})
#   %_low_memory_max_pool_offsets_to_indices : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_offsets_to_indices.default](args = (%getitem_3, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]), kwargs = {})
#   %max_pool2d_with_indices_backward : [num_users=1] = call_function[target=torch.ops.aten.max_pool2d_with_indices_backward.default](args = (%add_296, %relu, [3, 3], [2, 2], [1, 1], [1, 1], False, %_low_memory_max_pool_offsets_to_indices), kwargs = {})
triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_63 = async_compile.triton('triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_63(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 112)
    x2 = ((xindex // 7168) % 112)
    x3 = xindex // 802816
    x4 = ((xindex // 64) % 12544)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3584*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 200704*x3), None)
    tmp6 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3584*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 200704*x3), None).to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (x0 + 64*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3584*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 200704*x3), None)
    tmp17 = tl.load(in_ptr1 + (x0 + 64*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3584*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 200704*x3), None).to(tl.float32)
    tmp30 = tl.load(in_ptr0 + (x0 + 64*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3584*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 200704*x3), None)
    tmp35 = tl.load(in_ptr1 + (x0 + 64*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3584*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 200704*x3), None).to(tl.float32)
    tmp46 = tl.load(in_ptr0 + (x0 + 64*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3584*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 200704*x3), None)
    tmp51 = tl.load(in_ptr1 + (x0 + 64*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 3584*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))))) + 200704*x3), None).to(tl.float32)
    tmp1 = tl.full([XBLOCK], 9, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 9), "index out of bounds: 0 <= tmp4 < 9")
    tmp7 = (-113) + tmp4 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 109*(tmp4 // 3) + 224*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp8 = x4
    tmp9 = tmp7 == tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp6, tmp10)
    tmp13 = tmp12 + tmp1
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tl.device_assert((0 <= tmp15) & (tmp15 < 9), "index out of bounds: 0 <= tmp15 < 9")
    tmp18 = (-113) + tmp15 + 2*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 109*(tmp15 // 3) + 224*((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp19 = tmp18 == tmp8
    tmp20 = ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))
    tmp21 = ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))
    tmp22 = tmp20 < tmp21
    tmp23 = 1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))
    tmp24 = ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))
    tmp25 = tmp23 < tmp24
    tmp26 = tmp22 & tmp25
    tmp27 = tmp26 & tmp19
    tmp28 = tmp11 + tmp17
    tmp29 = tl.where(tmp27, tmp28, tmp11)
    tmp31 = tmp30 + tmp1
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.device_assert((0 <= tmp33) & (tmp33 < 9), "index out of bounds: 0 <= tmp33 < 9")
    tmp36 = (-113) + tmp33 + 2*((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 109*(tmp33 // 3) + 224*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp37 = tmp36 == tmp8
    tmp38 = 1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))
    tmp39 = tmp38 < tmp21
    tmp40 = ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))
    tmp41 = tmp40 < tmp24
    tmp42 = tmp39 & tmp41
    tmp43 = tmp42 & tmp37
    tmp44 = tmp29 + tmp35
    tmp45 = tl.where(tmp43, tmp44, tmp29)
    tmp47 = tmp46 + tmp1
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert((0 <= tmp49) & (tmp49 < 9), "index out of bounds: 0 <= tmp49 < 9")
    tmp52 = (-113) + tmp49 + 2*((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x1) // 2))) + (1 + ((1 + x1) // 2)) * ((1 + ((1 + x1) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x1 // 2)) + (x1 // 2) * ((x1 // 2) > (0)))))) + 109*(tmp49 // 3) + 224*((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) * ((1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0)))) <= ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56))))) + ((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) * (((-1) + ((56) * ((56) <= (1 + ((1 + x2) // 2))) + (1 + ((1 + x2) // 2)) * ((1 + ((1 + x2) // 2)) < (56)))) < (1 + ((0) * ((0) >= (x2 // 2)) + (x2 // 2) * ((x2 // 2) > (0))))))
    tmp53 = tmp52 == tmp8
    tmp54 = tmp39 & tmp25
    tmp55 = tmp54 & tmp53
    tmp56 = tmp45 + tmp51
    tmp57 = tl.where(tmp55, tmp56, tmp45)
    tl.store(out_ptr0 + (x7), tmp57, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vs/cvscvyx5wdbl7x7svfmnkvomsfyo7uirec4ippvagou7dfps7t6f.py
# Topologically Sorted Source Nodes: [scalar_tensor, x_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
#   x_1 => convert_element_type_2
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_48 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %where_48 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_48, %full_default, %max_pool2d_with_indices_backward), kwargs = {})
#   %convert_element_type_380 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_48, torch.float32), kwargs = {})
#   %sum_106 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_380, [0, 2, 3]), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution, torch.float32), kwargs = {})
#   %sub_261 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %unsqueeze_838), kwargs = {})
#   %mul_839 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_380, %sub_261), kwargs = {})
#   %sum_107 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_839, [0, 2, 3]), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_64 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_64', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2469462272, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (x0 + 64*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr2 + (x0 + 64*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask, tmp8, _tmp7)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fm/cfmfzgiz64veob6kopv22olulx4o33guhj7e2dyjzph3djw3vx7v.py
# Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_48 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %where_48 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_48, %full_default, %max_pool2d_with_indices_backward), kwargs = {})
#   %convert_element_type_380 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_48, torch.float32), kwargs = {})
#   %sum_106 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_380, [0, 2, 3]), kwargs = {})
triton_red_fused_native_batch_norm_backward_threshold_backward_65 = async_compile.triton('triton_red_fused_native_batch_norm_backward_threshold_backward_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 4096},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 803328, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_threshold_backward_65(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 3136
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/pq/cpqbbn32a6hcbctot6ucmysnofwfiwa4mfmeuzygnaovq2mmknxu.py
# Topologically Sorted Source Nodes: [scalar_tensor, x_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
#   x_1 => convert_element_type_2
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_48 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %where_48 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_48, %full_default, %max_pool2d_with_indices_backward), kwargs = {})
#   %convert_element_type_380 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_48, torch.float32), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution, torch.float32), kwargs = {})
#   %sub_261 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %unsqueeze_838), kwargs = {})
#   %mul_839 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_380, %sub_261), kwargs = {})
#   %sum_107 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_839, [0, 2, 3]), kwargs = {})
#   %mul_847 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_107, %squeeze_1), kwargs = {})
triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_66 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 4096},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 804096, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_66(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 3136
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/su/csuffwmljpbza422tuiajm2ohzinwb77sqnvxpis7jz2g55qklag.py
# Topologically Sorted Source Nodes: [scalar_tensor, x_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
# Source node to ATen node mapping:
#   scalar_tensor => full_default
#   x_1 => convert_element_type_2
# Graph fragment:
#   %full_default : [num_users=49] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %le_48 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
#   %where_48 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_48, %full_default, %max_pool2d_with_indices_backward), kwargs = {})
#   %convert_element_type_380 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_48, torch.float32), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution, torch.float32), kwargs = {})
#   %sub_261 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %unsqueeze_838), kwargs = {})
#   %mul_845 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_261, %unsqueeze_844), kwargs = {})
#   %sub_263 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_380, %mul_845), kwargs = {})
#   %sub_264 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_263, %unsqueeze_841), kwargs = {})
#   %mul_846 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_264, %unsqueeze_847), kwargs = {})
#   %convert_element_type_382 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_846, torch.bfloat16), kwargs = {})
#   %convolution_backward_52 : [num_users=1] = call_function[target=torch.ops.aten.convolution_backward.default](args = (%convert_element_type_382, %convert_element_type_1, %convert_element_type, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]), kwargs = {})
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_67 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_67', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4110419200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_67(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 411041792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 1.5570192920918366e-07
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/oo/coozpq2ffauxpto47dirvv4e552ryoom43psplkidxb7fmw4ryz3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {dtype: torch.float32})
triton_red_fused_sum_68 = async_compile.triton('triton_red_fused_sum_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 105600, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_68(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 400
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 100)
    x1 = xindex // 100
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 100*r0_2 + 12800*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/mm/cmmhfank6dhruqgq4x4jxn36znlwrmjstwzodwvjznzvqtngya6g.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {dtype: torch.float32})
triton_per_fused_sum_69 = async_compile.triton('triton_per_fused_sum_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2400, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_69(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 100
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
    tmp0 = tl.load(in_ptr0 + (x0 + 100*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fl/cflhd23qfd65dkdqgymlmk7blvdb3t7qgjn5em7grtuxg6s6byov.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_379 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_262, torch.float32), kwargs = {})
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


# kernel path: /tmp/torchinductor_yyu496/u2/cu2bgp5lfb5uqgnxht7fdq5eujl5xtpvo4j5bis7rkxpg25c3xxi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_383 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_265, torch.float32), kwargs = {})
triton_poi_fused__to_copy_71 = async_compile.triton('triton_poi_fused__to_copy_71', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_71', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 94080}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_71(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/de/cdegemvlrt57tzalzsrpk3e5an7hbhpawbdvctbyk7hxqwnenl3c.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_343 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_235, torch.float32), kwargs = {})
triton_poi_fused__to_copy_72 = async_compile.triton('triton_poi_fused__to_copy_72', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_72', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 163840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_72(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ws/cwszldokzftq3l5uyrlyhd2rcld3atzuysbkun52hy3ib25zrqvn.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_339 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_232, torch.float32), kwargs = {})
triton_poi_fused__to_copy_73 = async_compile.triton('triton_poi_fused__to_copy_73', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_73', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 327680}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_73(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/az/caz3rfunpwexkqn6nzbtzejqjap3twxbrial7xuymukr73csivdu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#    => constant_pad_nd_default
# Graph fragment:
#   %constant_pad_nd_default : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_2, [0, 0, 0, 4]), kwargs = {})
triton_poi_fused_mm_74 = async_compile.triton('triton_poi_fused_mm_74', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_74', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 319488}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_74(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_yyu496/t2/ct2i42pcokvp2hxrpk7odmgt7e4xfs5ntqc7zrnb6sirmhycnx76.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_347 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_238, torch.float32), kwargs = {})
triton_poi_fused__to_copy_75 = async_compile.triton('triton_poi_fused__to_copy_75', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_75', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 368640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_75(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/qu/cqu6padpniioc3nprkmtykkm7hmdj6lpgol6zbgew4trsommpy5m.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_291 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_196, torch.float32), kwargs = {})
triton_poi_fused__to_copy_76 = async_compile.triton('triton_poi_fused__to_copy_76', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_76', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 655360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_76(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ih/ciham72x2fpp6f5zdtto6gzmjkqoltv4ksvjhixoq4qyd56dhhsa.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_287 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_193, torch.float32), kwargs = {})
triton_poi_fused__to_copy_77 = async_compile.triton('triton_poi_fused__to_copy_77', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_77', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1310720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_77(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/be/cbecssydihtof4jzhnvasrlu6ooxgw6hlooo5ukrlweaya5nnkhg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_295 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_199, torch.float32), kwargs = {})
triton_poi_fused__to_copy_78 = async_compile.triton('triton_poi_fused__to_copy_78', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_78', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1474560}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_78(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ur/curs24vnbl4ipwclbvwenfbvg2yu5qmo2wnnu3x4tv7g2v4qtzks.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_170 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_tensor, torch.float32), kwargs = {})
triton_poi_fused__to_copy_79 = async_compile.triton('triton_poi_fused__to_copy_79', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_79', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_79(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5w/c5whrdbkgmk37zmt3oupu6fqgq2rgalvroae4kixqliybx7gc57p.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_215 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_139, torch.float32), kwargs = {})
triton_poi_fused__to_copy_80 = async_compile.triton('triton_poi_fused__to_copy_80', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_80', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2621440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_80(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lw/clwtdh4k5h4fkfkhhuzjbelvurdiyo5g2krtjqpt5jqjmwfgkaeq.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_211 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_136, torch.float32), kwargs = {})
triton_poi_fused__to_copy_81 = async_compile.triton('triton_poi_fused__to_copy_81', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_81', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5242880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_81(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/s4/cs4si7emiv7g5bn37y2ou7h62m7swmiaq5p2iseypeubn6xliu54.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_219 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_142, torch.float32), kwargs = {})
triton_poi_fused__to_copy_82 = async_compile.triton('triton_poi_fused__to_copy_82', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_82', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5898240}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_82(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ue/cuexyixymnopvojl5akxilfcuo6k74nxqrssxck4n7vovwukkxty.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_175 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_109, torch.float32), kwargs = {})
triton_poi_fused__to_copy_83 = async_compile.triton('triton_poi_fused__to_copy_83', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_83', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10485760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_83(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ul/cul4zat6audvdiqfkwh4l3nhmxq52o377uqt3n52q6dhkw3fyjs7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_199 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_127, torch.float32), kwargs = {})
triton_poi_fused__to_copy_84 = async_compile.triton('triton_poi_fused__to_copy_84', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_84', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20971520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_84(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/46/c46okaijabiojq66srtj2ylvcqzoqrmm4a67k5wcu5ovbzslcnsd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_179 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_112, torch.float32), kwargs = {})
triton_poi_fused__to_copy_85 = async_compile.triton('triton_poi_fused__to_copy_85', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_85', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 23592960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_85(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, primals_132, primals_138, primals_144, primals_150, primals_156, primals_162, primals_168, primals_174, primals_180, primals_186, primals_192, primals_198, primals_204, primals_210, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, primals_252, primals_258, primals_264, primals_270, primals_276, primals_282, primals_288, primals_294, primals_300, primals_306, primals_312, primals_318, convert_element_type, convert_element_type_1, convolution, squeeze_1, relu, getitem_2, getitem_3, convert_element_type_4, convolution_1, squeeze_4, relu_1, convert_element_type_7, convolution_2, squeeze_7, relu_2, convert_element_type_10, convolution_3, squeeze_10, convert_element_type_13, convolution_4, squeeze_13, relu_3, convert_element_type_16, convolution_5, squeeze_16, relu_4, convert_element_type_19, convolution_6, squeeze_19, relu_5, convert_element_type_22, convolution_7, squeeze_22, relu_6, convert_element_type_25, convolution_8, squeeze_25, relu_7, convert_element_type_28, convolution_9, squeeze_28, relu_8, convert_element_type_31, convolution_10, squeeze_31, relu_9, convert_element_type_34, convolution_11, squeeze_34, relu_10, convert_element_type_37, convolution_12, squeeze_37, relu_11, convert_element_type_40, convolution_13, squeeze_40, convert_element_type_43, convolution_14, squeeze_43, relu_12, convert_element_type_46, convolution_15, squeeze_46, relu_13, convert_element_type_49, convolution_16, squeeze_49, relu_14, convert_element_type_52, convolution_17, squeeze_52, relu_15, convert_element_type_55, convolution_18, squeeze_55, relu_16, convert_element_type_58, convolution_19, squeeze_58, relu_17, convert_element_type_61, convolution_20, squeeze_61, relu_18, convert_element_type_64, convolution_21, squeeze_64, relu_19, convert_element_type_67, convolution_22, squeeze_67, relu_20, convert_element_type_70, convolution_23, squeeze_70, relu_21, convert_element_type_73, convolution_24, squeeze_73, relu_22, convert_element_type_76, convolution_25, squeeze_76, relu_23, convert_element_type_79, convolution_26, squeeze_79, convert_element_type_82, convolution_27, squeeze_82, relu_24, convert_element_type_85, convolution_28, squeeze_85, relu_25, convert_element_type_88, convolution_29, squeeze_88, relu_26, convert_element_type_91, convolution_30, squeeze_91, relu_27, convert_element_type_94, convolution_31, squeeze_94, relu_28, convert_element_type_97, convolution_32, squeeze_97, relu_29, convert_element_type_100, convolution_33, squeeze_100, relu_30, convert_element_type_103, convolution_34, squeeze_103, relu_31, convert_element_type_106, convolution_35, squeeze_106, relu_32, convert_element_type_109, convolution_36, squeeze_109, relu_33, convert_element_type_112, convolution_37, squeeze_112, relu_34, convert_element_type_115, convolution_38, squeeze_115, relu_35, convert_element_type_118, convolution_39, squeeze_118, relu_36, convert_element_type_121, convolution_40, squeeze_121, relu_37, convert_element_type_124, convolution_41, squeeze_124, relu_38, convert_element_type_127, convolution_42, squeeze_127, relu_39, convert_element_type_130, convolution_43, squeeze_130, relu_40, convert_element_type_133, convolution_44, squeeze_133, relu_41, convert_element_type_136, convolution_45, squeeze_136, convert_element_type_139, convolution_46, squeeze_139, relu_42, convert_element_type_142, convolution_47, squeeze_142, relu_43, convert_element_type_145, convolution_48, squeeze_145, relu_44, convert_element_type_148, convolution_49, squeeze_148, relu_45, convert_element_type_151, convolution_50, squeeze_151, relu_46, convert_element_type_154, convolution_51, squeeze_154, relu_47, convert_element_type_157, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1 = args
    args.clear()
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_102, (128, ), (1, ))
    assert_size_stride(primals_108, (512, ), (1, ))
    assert_size_stride(primals_114, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_150, (256, ), (1, ))
    assert_size_stride(primals_156, (256, ), (1, ))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_168, (1024, ), (1, ))
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_180, (256, ), (1, ))
    assert_size_stride(primals_186, (1024, ), (1, ))
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_204, (1024, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_228, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_246, (256, ), (1, ))
    assert_size_stride(primals_252, (256, ), (1, ))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_276, (2048, ), (1, ))
    assert_size_stride(primals_282, (2048, ), (1, ))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_300, (2048, ), (1, ))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_318, (2048, ), (1, ))
    assert_size_stride(convert_element_type, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(convert_element_type_1, (512, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (512, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (512, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(getitem_2, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(getitem_3, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_4, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_1, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu_1, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_7, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_2, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_10, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_3, (512, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_10, (256, ), (1, ))
    assert_size_stride(convert_element_type_13, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_4, (512, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_13, (256, ), (1, ))
    assert_size_stride(relu_3, (512, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convert_element_type_16, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_5, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_16, (64, ), (1, ))
    assert_size_stride(relu_4, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_19, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_6, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_19, (64, ), (1, ))
    assert_size_stride(relu_5, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_22, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_7, (512, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_22, (256, ), (1, ))
    assert_size_stride(relu_6, (512, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convert_element_type_25, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_8, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(relu_7, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_28, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(convolution_9, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(squeeze_28, (64, ), (1, ))
    assert_size_stride(relu_8, (512, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convert_element_type_31, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_10, (512, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(relu_9, (512, 256, 56, 56), (802816, 1, 14336, 256))
    assert_size_stride(convert_element_type_34, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_11, (512, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(relu_10, (512, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(convert_element_type_37, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_12, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_37, (128, ), (1, ))
    assert_size_stride(relu_11, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convert_element_type_40, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_13, (512, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_40, (512, ), (1, ))
    assert_size_stride(convert_element_type_43, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_14, (512, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_43, (512, ), (1, ))
    assert_size_stride(relu_12, (512, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convert_element_type_46, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_15, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(relu_13, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convert_element_type_49, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_16, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_49, (128, ), (1, ))
    assert_size_stride(relu_14, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convert_element_type_52, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_17, (512, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_52, (512, ), (1, ))
    assert_size_stride(relu_15, (512, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convert_element_type_55, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_18, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_55, (128, ), (1, ))
    assert_size_stride(relu_16, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convert_element_type_58, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_19, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(relu_17, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convert_element_type_61, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_20, (512, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_61, (512, ), (1, ))
    assert_size_stride(relu_18, (512, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convert_element_type_64, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_21, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(relu_19, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convert_element_type_67, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(convolution_22, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(squeeze_67, (128, ), (1, ))
    assert_size_stride(relu_20, (512, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(convert_element_type_70, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(convolution_23, (512, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(squeeze_70, (512, ), (1, ))
    assert_size_stride(relu_21, (512, 512, 28, 28), (401408, 1, 14336, 512))
    assert_size_stride(convert_element_type_73, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_24, (512, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(relu_22, (512, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(convert_element_type_76, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convolution_25, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_76, (256, ), (1, ))
    assert_size_stride(relu_23, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_79, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_26, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_79, (1024, ), (1, ))
    assert_size_stride(convert_element_type_82, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_27, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_82, (1024, ), (1, ))
    assert_size_stride(relu_24, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convert_element_type_85, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convolution_28, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(relu_25, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_88, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convolution_29, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_88, (256, ), (1, ))
    assert_size_stride(relu_26, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_91, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_30, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_91, (1024, ), (1, ))
    assert_size_stride(relu_27, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convert_element_type_94, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convolution_31, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_94, (256, ), (1, ))
    assert_size_stride(relu_28, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_97, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convolution_32, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_97, (256, ), (1, ))
    assert_size_stride(relu_29, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_100, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_33, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_100, (1024, ), (1, ))
    assert_size_stride(relu_30, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convert_element_type_103, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convolution_34, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_103, (256, ), (1, ))
    assert_size_stride(relu_31, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_106, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convolution_35, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_106, (256, ), (1, ))
    assert_size_stride(relu_32, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_109, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_36, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_109, (1024, ), (1, ))
    assert_size_stride(relu_33, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convert_element_type_112, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convolution_37, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_112, (256, ), (1, ))
    assert_size_stride(relu_34, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_115, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convolution_38, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(relu_35, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_118, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_39, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_118, (1024, ), (1, ))
    assert_size_stride(relu_36, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convert_element_type_121, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convolution_40, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_121, (256, ), (1, ))
    assert_size_stride(relu_37, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_124, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(convolution_41, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(squeeze_124, (256, ), (1, ))
    assert_size_stride(relu_38, (512, 256, 14, 14), (50176, 1, 3584, 256))
    assert_size_stride(convert_element_type_127, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(convolution_42, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(squeeze_127, (1024, ), (1, ))
    assert_size_stride(relu_39, (512, 1024, 14, 14), (200704, 1, 14336, 1024))
    assert_size_stride(convert_element_type_130, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convolution_43, (512, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(squeeze_130, (512, ), (1, ))
    assert_size_stride(relu_40, (512, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(convert_element_type_133, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(convolution_44, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_133, (512, ), (1, ))
    assert_size_stride(relu_41, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convert_element_type_136, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_45, (512, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_136, (2048, ), (1, ))
    assert_size_stride(convert_element_type_139, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(convolution_46, (512, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_139, (2048, ), (1, ))
    assert_size_stride(relu_42, (512, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convert_element_type_142, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(convolution_47, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_142, (512, ), (1, ))
    assert_size_stride(relu_43, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convert_element_type_145, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(convolution_48, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_145, (512, ), (1, ))
    assert_size_stride(relu_44, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convert_element_type_148, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_49, (512, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_148, (2048, ), (1, ))
    assert_size_stride(relu_45, (512, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(convert_element_type_151, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(convolution_50, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_151, (512, ), (1, ))
    assert_size_stride(relu_46, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convert_element_type_154, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(convolution_51, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(squeeze_154, (512, ), (1, ))
    assert_size_stride(relu_47, (512, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(convert_element_type_157, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(convolution_52, (512, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(squeeze_157, (2048, ), (1, ))
    assert_size_stride(view, (512, 2048), (2048, 1))
    assert_size_stride(permute_1, (100, 2048), (2048, 1))
    assert_size_stride(le, (512, 2048, 7, 7), (100352, 1, 14336, 2048))
    assert_size_stride(unsqueeze_214, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_226, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_238, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_250, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_514, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_586, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_682, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_706, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_718, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_730, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_754, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_766, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_778, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_790, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_802, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_826, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_838, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (512, 100), (100, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((512, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf6 = empty_strided_cuda((2048, 64), (1, 2048), torch.float32)
        buf8 = empty_strided_cuda((2048, 64), (1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_157], Original ATen: [aten.div, aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_0.run(le, buf0, convolution_52, unsqueeze_214, buf6, buf8, 131072, 392, stream=stream0)
        buf7 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.div, aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(buf6, buf7, 2048, 64, stream=stream0)
        buf9 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf10 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_157], Original ATen: [aten.div, aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_2.run(buf8, squeeze_157, buf9, buf10, 2048, 64, stream=stream0)
        buf11 = convolution_52; del convolution_52  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_157], Original ATen: [aten.div, aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_3.run(buf11, le, buf0, unsqueeze_214, buf9, squeeze_157, buf7, primals_318, 51380224, stream=stream0)
        del primals_318
        del squeeze_157
        del unsqueeze_214
        # Topologically Sorted Source Nodes: [scalar_tensor, out_157], Original ATen: [aten.div, aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf12 = torch.ops.aten.convolution_backward.default(buf11, relu_47, convert_element_type_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_157
        buf13 = buf12[0]
        assert_size_stride(buf13, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf13, 16, 'torch.ops.aten.convolution_backward.default')
        buf14 = buf12[1]
        assert_size_stride(buf14, (2048, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf14, 16, 'torch.ops.aten.convolution_backward.default')
        del buf12
        buf16 = empty_strided_cuda((512, 196), (1, 512), torch.float32)
        buf18 = empty_strided_cuda((512, 196), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_154], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4.run(relu_47, buf13, convolution_51, unsqueeze_226, buf16, buf18, 100352, 128, stream=stream0)
        buf17 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf16, buf17, 512, 196, stream=stream0)
        buf19 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf20 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_154], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf18, squeeze_154, buf19, buf20, 512, 196, stream=stream0)
        buf21 = relu_47; del relu_47  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_154], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_7.run(buf21, buf13, convolution_51, unsqueeze_226, buf19, squeeze_154, buf17, primals_312, 12845056, stream=stream0)
        del buf13
        del convolution_51
        del primals_312
        del squeeze_154
        del unsqueeze_226
        # Topologically Sorted Source Nodes: [scalar_tensor, out_154], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf22 = torch.ops.aten.convolution_backward.default(buf21, relu_46, convert_element_type_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf21
        del convert_element_type_154
        buf23 = buf22[0]
        assert_size_stride(buf23, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf23, 16, 'torch.ops.aten.convolution_backward.default')
        buf24 = buf22[1]
        assert_size_stride(buf24, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf24, 16, 'torch.ops.aten.convolution_backward.default')
        del buf22
        buf26 = buf18; del buf18  # reuse
        buf28 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_151], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4.run(relu_46, buf23, convolution_50, unsqueeze_238, buf26, buf28, 100352, 128, stream=stream0)
        buf27 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf26, buf27, 512, 196, stream=stream0)
        buf29 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf30 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_151], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf28, squeeze_151, buf29, buf30, 512, 196, stream=stream0)
        buf31 = relu_46; del relu_46  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_151], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_7.run(buf31, buf23, convolution_50, unsqueeze_238, buf29, squeeze_151, buf27, primals_306, 12845056, stream=stream0)
        del buf23
        del convolution_50
        del primals_306
        del squeeze_151
        del unsqueeze_238
        # Topologically Sorted Source Nodes: [scalar_tensor, out_151], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf32 = torch.ops.aten.convolution_backward.default(buf31, relu_45, convert_element_type_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf31
        del convert_element_type_151
        buf33 = buf32[0]
        assert_size_stride(buf33, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf33, 16, 'torch.ops.aten.convolution_backward.default')
        buf34 = buf32[1]
        assert_size_stride(buf34, (512, 2048, 1, 1), (2048, 1, 2048, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf34, 16, 'torch.ops.aten.convolution_backward.default')
        del buf32
        buf36 = buf8; del buf8  # reuse
        buf38 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_147], Original ATen: [aten.div, aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_div_native_batch_norm_backward_threshold_backward_8.run(relu_45, le, buf0, buf33, convolution_49, unsqueeze_250, buf36, buf38, 131072, 392, stream=stream0)
        buf37 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.div, aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(buf36, buf37, 2048, 64, stream=stream0)
        buf39 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf41 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_147], Original ATen: [aten.div, aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_2.run(buf38, squeeze_148, buf39, buf41, 2048, 64, stream=stream0)
        buf42 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_147], Original ATen: [aten.div, aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_9.run(relu_45, le, buf0, buf33, convolution_49, unsqueeze_250, buf39, squeeze_148, buf37, primals_300, buf42, 51380224, stream=stream0)
        del convolution_49
        del primals_300
        del squeeze_148
        del unsqueeze_250
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        buf43 = torch.ops.aten.convolution_backward.default(buf42, relu_44, convert_element_type_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf42
        del convert_element_type_148
        buf44 = buf43[0]
        assert_size_stride(buf44, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf44, 16, 'torch.ops.aten.convolution_backward.default')
        buf45 = buf43[1]
        assert_size_stride(buf45, (2048, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf45, 16, 'torch.ops.aten.convolution_backward.default')
        del buf43
        buf47 = buf28; del buf28  # reuse
        buf49 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_144], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4.run(relu_44, buf44, convolution_48, unsqueeze_262, buf47, buf49, 100352, 128, stream=stream0)
        buf48 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf47, buf48, 512, 196, stream=stream0)
        buf50 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf51 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_144], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf49, squeeze_145, buf50, buf51, 512, 196, stream=stream0)
        buf52 = relu_44; del relu_44  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_144], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_7.run(buf52, buf44, convolution_48, unsqueeze_262, buf50, squeeze_145, buf48, primals_294, 12845056, stream=stream0)
        del buf44
        del convolution_48
        del primals_294
        del squeeze_145
        del unsqueeze_262
        # Topologically Sorted Source Nodes: [scalar_tensor, out_144], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf53 = torch.ops.aten.convolution_backward.default(buf52, relu_43, convert_element_type_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf52
        del convert_element_type_145
        buf54 = buf53[0]
        assert_size_stride(buf54, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf54, 16, 'torch.ops.aten.convolution_backward.default')
        buf55 = buf53[1]
        assert_size_stride(buf55, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf55, 16, 'torch.ops.aten.convolution_backward.default')
        del buf53
        buf57 = buf49; del buf49  # reuse
        buf59 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_141], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4.run(relu_43, buf54, convolution_47, unsqueeze_274, buf57, buf59, 100352, 128, stream=stream0)
        buf58 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf57, buf58, 512, 196, stream=stream0)
        buf60 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf61 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_141], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf59, squeeze_142, buf60, buf61, 512, 196, stream=stream0)
        buf62 = relu_43; del relu_43  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_141], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_7.run(buf62, buf54, convolution_47, unsqueeze_274, buf60, squeeze_142, buf58, primals_288, 12845056, stream=stream0)
        del buf54
        del convolution_47
        del primals_288
        del squeeze_142
        del unsqueeze_274
        # Topologically Sorted Source Nodes: [scalar_tensor, out_141], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf63 = torch.ops.aten.convolution_backward.default(buf62, relu_42, convert_element_type_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf62
        del convert_element_type_142
        buf64 = buf63[0]
        assert_size_stride(buf64, (512, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf64, 16, 'torch.ops.aten.convolution_backward.default')
        buf65 = buf63[1]
        assert_size_stride(buf65, (512, 2048, 1, 1), (2048, 1, 2048, 2048), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf65, 16, 'torch.ops.aten.convolution_backward.default')
        del buf63
        buf67 = empty_strided_cuda((512, 2048, 7, 7), (100352, 1, 14336, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.div, aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_native_batch_norm_backward_threshold_backward_10.run(relu_42, relu_45, le, buf0, buf33, buf64, buf67, 51380224, stream=stream0)
        del buf0
        del buf33
        del buf64
        del le
        del relu_42
        del relu_45
        buf68 = buf38; del buf38  # reuse
        buf70 = buf36; del buf36  # reuse
        buf78 = empty_strided_cuda((2048, 64), (1, 2048), torch.float32)
        buf80 = empty_strided_cuda((2048, 64), (1, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, out_137], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_11.run(buf67, convolution_46, unsqueeze_286, convolution_45, unsqueeze_298, buf68, buf70, buf78, buf80, 131072, 392, stream=stream0)
        buf69 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(buf68, buf69, 2048, 64, stream=stream0)
        buf79 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_1.run(buf78, buf79, 2048, 64, stream=stream0)
        buf71 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf72 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_2.run(buf70, squeeze_139, buf71, buf72, 2048, 64, stream=stream0)
        buf81 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf82 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_functional_div_native_batch_norm_backward_threshold_backward_2.run(buf80, squeeze_136, buf81, buf82, 2048, 64, stream=stream0)
        buf73 = convolution_46; del convolution_46  # reuse
        buf83 = convolution_45; del convolution_45  # reuse
        # Topologically Sorted Source Nodes: [input_8, out_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_12.run(buf73, buf83, buf67, unsqueeze_286, buf71, squeeze_139, buf69, primals_282, unsqueeze_298, buf81, squeeze_136, buf79, primals_276, 51380224, stream=stream0)
        del buf67
        del buf71
        del buf81
        del primals_276
        del primals_282
        del squeeze_136
        del squeeze_139
        del unsqueeze_286
        del unsqueeze_298
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        buf74 = torch.ops.aten.convolution_backward.default(buf73, relu_39, convert_element_type_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf73
        del convert_element_type_139
        buf75 = buf74[0]
        assert_size_stride(buf75, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf75, 16, 'torch.ops.aten.convolution_backward.default')
        buf76 = buf74[1]
        assert_size_stride(buf76, (2048, 1024, 1, 1), (1024, 1, 1024, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf76, 16, 'torch.ops.aten.convolution_backward.default')
        del buf74
        # Topologically Sorted Source Nodes: [out_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        buf84 = torch.ops.aten.convolution_backward.default(buf83, relu_41, convert_element_type_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf83
        del convert_element_type_136
        buf85 = buf84[0]
        assert_size_stride(buf85, (512, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf85, 16, 'torch.ops.aten.convolution_backward.default')
        buf86 = buf84[1]
        assert_size_stride(buf86, (2048, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf86, 16, 'torch.ops.aten.convolution_backward.default')
        del buf84
        buf88 = buf59; del buf59  # reuse
        buf90 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_134], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_4.run(relu_41, buf85, convolution_44, unsqueeze_310, buf88, buf90, 100352, 128, stream=stream0)
        buf89 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf88, buf89, 512, 196, stream=stream0)
        buf91 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf92 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_134], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf90, squeeze_133, buf91, buf92, 512, 196, stream=stream0)
        buf93 = relu_41; del relu_41  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_134], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_7.run(buf93, buf85, convolution_44, unsqueeze_310, buf91, squeeze_133, buf89, primals_270, 12845056, stream=stream0)
        del buf85
        del convolution_44
        del primals_270
        del squeeze_133
        del unsqueeze_310
        # Topologically Sorted Source Nodes: [scalar_tensor, out_134], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf94 = torch.ops.aten.convolution_backward.default(buf93, relu_40, convert_element_type_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf93
        del convert_element_type_133
        buf95 = buf94[0]
        assert_size_stride(buf95, (512, 512, 14, 14), (100352, 1, 7168, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf95, 16, 'torch.ops.aten.convolution_backward.default')
        buf96 = buf94[1]
        assert_size_stride(buf96, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf96, 16, 'torch.ops.aten.convolution_backward.default')
        del buf94
        buf98 = reinterpret_tensor(buf80, (512, 256), (1, 512), 0); del buf80  # reuse
        buf100 = reinterpret_tensor(buf70, (512, 256), (1, 512), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_131], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_13.run(relu_40, buf95, convolution_43, unsqueeze_322, buf98, buf100, 131072, 392, stream=stream0)
        buf99 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_14.run(buf98, buf99, 512, 256, stream=stream0)
        buf101 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf102 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_131], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_15.run(buf100, squeeze_130, buf101, buf102, 512, 256, stream=stream0)
        buf103 = relu_40; del relu_40  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_131], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_16.run(buf103, buf95, convolution_43, unsqueeze_322, buf101, squeeze_130, buf99, primals_264, 51380224, stream=stream0)
        del buf95
        del convolution_43
        del primals_264
        del squeeze_130
        del unsqueeze_322
        # Topologically Sorted Source Nodes: [scalar_tensor, out_131], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf104 = torch.ops.aten.convolution_backward.default(buf103, relu_39, convert_element_type_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf103
        del convert_element_type_130
        buf105 = buf104[0]
        assert_size_stride(buf105, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf105, 16, 'torch.ops.aten.convolution_backward.default')
        buf106 = buf104[1]
        assert_size_stride(buf106, (512, 1024, 1, 1), (1024, 1, 1024, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf106, 16, 'torch.ops.aten.convolution_backward.default')
        del buf104
        buf108 = reinterpret_tensor(buf100, (1024, 128), (1, 1024), 0); del buf100  # reuse
        buf110 = reinterpret_tensor(buf98, (1024, 128), (1, 1024), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_127], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_17.run(relu_39, buf75, buf105, convolution_42, unsqueeze_334, buf108, buf110, 131072, 784, stream=stream0)
        buf109 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(buf108, buf109, 1024, 128, stream=stream0)
        buf111 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf113 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_127], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19.run(buf110, squeeze_127, buf111, buf113, 1024, 128, stream=stream0)
        buf114 = empty_strided_cuda((512, 1024, 14, 14), (200704, 1, 14336, 1024), torch.bfloat16)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_127], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_20.run(relu_39, buf75, buf105, convolution_42, unsqueeze_334, buf111, squeeze_127, buf109, primals_258, buf114, 102760448, stream=stream0)
        del convolution_42
        del primals_258
        del squeeze_127
        del unsqueeze_334
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        buf115 = torch.ops.aten.convolution_backward.default(buf114, relu_38, convert_element_type_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf114
        del convert_element_type_127
        buf116 = buf115[0]
        assert_size_stride(buf116, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf116, 16, 'torch.ops.aten.convolution_backward.default')
        buf117 = buf115[1]
        assert_size_stride(buf117, (1024, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf117, 16, 'torch.ops.aten.convolution_backward.default')
        del buf115
        buf119 = reinterpret_tensor(buf110, (256, 512), (1, 256), 0); del buf110  # reuse
        buf121 = reinterpret_tensor(buf108, (256, 512), (1, 256), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_124], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_38, buf116, convolution_41, unsqueeze_346, buf119, buf121, 131072, 196, stream=stream0)
        buf120 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf119, buf120, 256, 512, stream=stream0)
        buf122 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf123 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_124], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf121, squeeze_124, buf122, buf123, 256, 512, stream=stream0)
        buf124 = relu_38; del relu_38  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_124], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf124, buf116, convolution_41, unsqueeze_346, buf122, squeeze_124, buf120, primals_252, 25690112, stream=stream0)
        del buf116
        del convolution_41
        del primals_252
        del squeeze_124
        del unsqueeze_346
        # Topologically Sorted Source Nodes: [scalar_tensor, out_124], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf125 = torch.ops.aten.convolution_backward.default(buf124, relu_37, convert_element_type_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf124
        del convert_element_type_124
        buf126 = buf125[0]
        assert_size_stride(buf126, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf126, 16, 'torch.ops.aten.convolution_backward.default')
        buf127 = buf125[1]
        assert_size_stride(buf127, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf127, 16, 'torch.ops.aten.convolution_backward.default')
        del buf125
        buf129 = buf121; del buf121  # reuse
        buf131 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_121], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_37, buf126, convolution_40, unsqueeze_358, buf129, buf131, 131072, 196, stream=stream0)
        buf130 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf129, buf130, 256, 512, stream=stream0)
        buf132 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf133 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_121], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf131, squeeze_121, buf132, buf133, 256, 512, stream=stream0)
        buf134 = relu_37; del relu_37  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_121], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf134, buf126, convolution_40, unsqueeze_358, buf132, squeeze_121, buf130, primals_246, 25690112, stream=stream0)
        del buf126
        del convolution_40
        del primals_246
        del squeeze_121
        del unsqueeze_358
        # Topologically Sorted Source Nodes: [scalar_tensor, out_121], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf135 = torch.ops.aten.convolution_backward.default(buf134, relu_36, convert_element_type_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf134
        del convert_element_type_121
        buf136 = buf135[0]
        assert_size_stride(buf136, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf136, 16, 'torch.ops.aten.convolution_backward.default')
        buf137 = buf135[1]
        assert_size_stride(buf137, (256, 1024, 1, 1), (1024, 1, 1024, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf137, 16, 'torch.ops.aten.convolution_backward.default')
        del buf135
        buf139 = relu_36; del relu_36  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_threshold_backward_25.run(buf139, relu_39, buf75, buf105, buf136, 102760448, stream=stream0)
        del buf105
        del buf136
        del buf75
        del relu_39
        buf140 = reinterpret_tensor(buf131, (1024, 128), (1, 1024), 0); del buf131  # reuse
        buf142 = reinterpret_tensor(buf129, (1024, 128), (1, 1024), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [out_117], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_26.run(buf139, convolution_39, unsqueeze_370, buf140, buf142, 131072, 784, stream=stream0)
        buf141 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(buf140, buf141, 1024, 128, stream=stream0)
        buf143 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf144 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_117], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19.run(buf142, squeeze_118, buf143, buf144, 1024, 128, stream=stream0)
        buf145 = convolution_39; del convolution_39  # reuse
        # Topologically Sorted Source Nodes: [out_117], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_27.run(buf145, buf139, unsqueeze_370, buf143, squeeze_118, buf141, primals_240, 102760448, stream=stream0)
        del primals_240
        del squeeze_118
        del unsqueeze_370
        # Topologically Sorted Source Nodes: [out_117], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf146 = torch.ops.aten.convolution_backward.default(buf145, relu_35, convert_element_type_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_118
        buf147 = buf146[0]
        assert_size_stride(buf147, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf147, 16, 'torch.ops.aten.convolution_backward.default')
        buf148 = buf146[1]
        assert_size_stride(buf148, (1024, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf148, 16, 'torch.ops.aten.convolution_backward.default')
        del buf146
        buf150 = reinterpret_tensor(buf142, (256, 512), (1, 256), 0); del buf142  # reuse
        buf152 = reinterpret_tensor(buf140, (256, 512), (1, 256), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_114], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_35, buf147, convolution_38, unsqueeze_382, buf150, buf152, 131072, 196, stream=stream0)
        buf151 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf150, buf151, 256, 512, stream=stream0)
        buf153 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf154 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_114], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf152, squeeze_115, buf153, buf154, 256, 512, stream=stream0)
        buf155 = relu_35; del relu_35  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_114], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf155, buf147, convolution_38, unsqueeze_382, buf153, squeeze_115, buf151, primals_234, 25690112, stream=stream0)
        del buf147
        del convolution_38
        del primals_234
        del squeeze_115
        del unsqueeze_382
        # Topologically Sorted Source Nodes: [scalar_tensor, out_114], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf156 = torch.ops.aten.convolution_backward.default(buf155, relu_34, convert_element_type_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf155
        del convert_element_type_115
        buf157 = buf156[0]
        assert_size_stride(buf157, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf157, 16, 'torch.ops.aten.convolution_backward.default')
        buf158 = buf156[1]
        assert_size_stride(buf158, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf158, 16, 'torch.ops.aten.convolution_backward.default')
        del buf156
        buf160 = buf152; del buf152  # reuse
        buf162 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_111], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_34, buf157, convolution_37, unsqueeze_394, buf160, buf162, 131072, 196, stream=stream0)
        buf161 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf160, buf161, 256, 512, stream=stream0)
        buf163 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf164 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_111], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf162, squeeze_112, buf163, buf164, 256, 512, stream=stream0)
        buf165 = relu_34; del relu_34  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_111], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf165, buf157, convolution_37, unsqueeze_394, buf163, squeeze_112, buf161, primals_228, 25690112, stream=stream0)
        del buf157
        del convolution_37
        del primals_228
        del squeeze_112
        del unsqueeze_394
        # Topologically Sorted Source Nodes: [scalar_tensor, out_111], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf166 = torch.ops.aten.convolution_backward.default(buf165, relu_33, convert_element_type_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf165
        del convert_element_type_112
        buf167 = buf166[0]
        assert_size_stride(buf167, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf167, 16, 'torch.ops.aten.convolution_backward.default')
        buf168 = buf166[1]
        assert_size_stride(buf168, (256, 1024, 1, 1), (1024, 1, 1024, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf168, 16, 'torch.ops.aten.convolution_backward.default')
        del buf166
        buf170 = reinterpret_tensor(buf162, (1024, 128), (1, 1024), 0); del buf162  # reuse
        buf172 = reinterpret_tensor(buf160, (1024, 128), (1, 1024), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_107], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_17.run(relu_33, buf139, buf167, convolution_36, unsqueeze_406, buf170, buf172, 131072, 784, stream=stream0)
        buf171 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(buf170, buf171, 1024, 128, stream=stream0)
        buf173 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf175 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_107], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19.run(buf172, squeeze_109, buf173, buf175, 1024, 128, stream=stream0)
        buf176 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_107], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_20.run(relu_33, buf139, buf167, convolution_36, unsqueeze_406, buf173, squeeze_109, buf171, primals_222, buf176, 102760448, stream=stream0)
        del convolution_36
        del primals_222
        del squeeze_109
        del unsqueeze_406
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        buf177 = torch.ops.aten.convolution_backward.default(buf176, relu_32, convert_element_type_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf176
        del convert_element_type_109
        buf178 = buf177[0]
        assert_size_stride(buf178, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf178, 16, 'torch.ops.aten.convolution_backward.default')
        buf179 = buf177[1]
        assert_size_stride(buf179, (1024, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf179, 16, 'torch.ops.aten.convolution_backward.default')
        del buf177
        buf181 = reinterpret_tensor(buf172, (256, 512), (1, 256), 0); del buf172  # reuse
        buf183 = reinterpret_tensor(buf170, (256, 512), (1, 256), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_104], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_32, buf178, convolution_35, unsqueeze_418, buf181, buf183, 131072, 196, stream=stream0)
        buf182 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf181, buf182, 256, 512, stream=stream0)
        buf184 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf185 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_104], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf183, squeeze_106, buf184, buf185, 256, 512, stream=stream0)
        buf186 = relu_32; del relu_32  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_104], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf186, buf178, convolution_35, unsqueeze_418, buf184, squeeze_106, buf182, primals_216, 25690112, stream=stream0)
        del buf178
        del convolution_35
        del primals_216
        del squeeze_106
        del unsqueeze_418
        # Topologically Sorted Source Nodes: [scalar_tensor, out_104], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf187 = torch.ops.aten.convolution_backward.default(buf186, relu_31, convert_element_type_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf186
        del convert_element_type_106
        buf188 = buf187[0]
        assert_size_stride(buf188, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf188, 16, 'torch.ops.aten.convolution_backward.default')
        buf189 = buf187[1]
        assert_size_stride(buf189, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf189, 16, 'torch.ops.aten.convolution_backward.default')
        del buf187
        buf191 = buf183; del buf183  # reuse
        buf193 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_101], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_31, buf188, convolution_34, unsqueeze_430, buf191, buf193, 131072, 196, stream=stream0)
        buf192 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf191, buf192, 256, 512, stream=stream0)
        buf194 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf195 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_101], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf193, squeeze_103, buf194, buf195, 256, 512, stream=stream0)
        buf196 = relu_31; del relu_31  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_101], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf196, buf188, convolution_34, unsqueeze_430, buf194, squeeze_103, buf192, primals_210, 25690112, stream=stream0)
        del buf188
        del convolution_34
        del primals_210
        del squeeze_103
        del unsqueeze_430
        # Topologically Sorted Source Nodes: [scalar_tensor, out_101], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf197 = torch.ops.aten.convolution_backward.default(buf196, relu_30, convert_element_type_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf196
        del convert_element_type_103
        buf198 = buf197[0]
        assert_size_stride(buf198, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf198, 16, 'torch.ops.aten.convolution_backward.default')
        buf199 = buf197[1]
        assert_size_stride(buf199, (256, 1024, 1, 1), (1024, 1, 1024, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf199, 16, 'torch.ops.aten.convolution_backward.default')
        del buf197
        buf201 = relu_30; del relu_30  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_threshold_backward_25.run(buf201, relu_33, buf139, buf167, buf198, 102760448, stream=stream0)
        del buf139
        del buf167
        del buf198
        del relu_33
        buf202 = reinterpret_tensor(buf193, (1024, 128), (1, 1024), 0); del buf193  # reuse
        buf204 = reinterpret_tensor(buf191, (1024, 128), (1, 1024), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [out_97], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_26.run(buf201, convolution_33, unsqueeze_442, buf202, buf204, 131072, 784, stream=stream0)
        buf203 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(buf202, buf203, 1024, 128, stream=stream0)
        buf205 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf206 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_97], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19.run(buf204, squeeze_100, buf205, buf206, 1024, 128, stream=stream0)
        buf207 = convolution_33; del convolution_33  # reuse
        # Topologically Sorted Source Nodes: [out_97], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_27.run(buf207, buf201, unsqueeze_442, buf205, squeeze_100, buf203, primals_204, 102760448, stream=stream0)
        del primals_204
        del squeeze_100
        del unsqueeze_442
        # Topologically Sorted Source Nodes: [out_97], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf208 = torch.ops.aten.convolution_backward.default(buf207, relu_29, convert_element_type_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_100
        buf209 = buf208[0]
        assert_size_stride(buf209, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf209, 16, 'torch.ops.aten.convolution_backward.default')
        buf210 = buf208[1]
        assert_size_stride(buf210, (1024, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf210, 16, 'torch.ops.aten.convolution_backward.default')
        del buf208
        buf212 = reinterpret_tensor(buf204, (256, 512), (1, 256), 0); del buf204  # reuse
        buf214 = reinterpret_tensor(buf202, (256, 512), (1, 256), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_94], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_29, buf209, convolution_32, unsqueeze_454, buf212, buf214, 131072, 196, stream=stream0)
        buf213 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf212, buf213, 256, 512, stream=stream0)
        buf215 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf216 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_94], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf214, squeeze_97, buf215, buf216, 256, 512, stream=stream0)
        buf217 = relu_29; del relu_29  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_94], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf217, buf209, convolution_32, unsqueeze_454, buf215, squeeze_97, buf213, primals_198, 25690112, stream=stream0)
        del buf209
        del convolution_32
        del primals_198
        del squeeze_97
        del unsqueeze_454
        # Topologically Sorted Source Nodes: [scalar_tensor, out_94], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf218 = torch.ops.aten.convolution_backward.default(buf217, relu_28, convert_element_type_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf217
        del convert_element_type_97
        buf219 = buf218[0]
        assert_size_stride(buf219, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf219, 16, 'torch.ops.aten.convolution_backward.default')
        buf220 = buf218[1]
        assert_size_stride(buf220, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf220, 16, 'torch.ops.aten.convolution_backward.default')
        del buf218
        buf222 = buf214; del buf214  # reuse
        buf224 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_91], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_28, buf219, convolution_31, unsqueeze_466, buf222, buf224, 131072, 196, stream=stream0)
        buf223 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf222, buf223, 256, 512, stream=stream0)
        buf225 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf226 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_91], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf224, squeeze_94, buf225, buf226, 256, 512, stream=stream0)
        buf227 = relu_28; del relu_28  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_91], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf227, buf219, convolution_31, unsqueeze_466, buf225, squeeze_94, buf223, primals_192, 25690112, stream=stream0)
        del buf219
        del convolution_31
        del primals_192
        del squeeze_94
        del unsqueeze_466
        # Topologically Sorted Source Nodes: [scalar_tensor, out_91], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf228 = torch.ops.aten.convolution_backward.default(buf227, relu_27, convert_element_type_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf227
        del convert_element_type_94
        buf229 = buf228[0]
        assert_size_stride(buf229, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf229, 16, 'torch.ops.aten.convolution_backward.default')
        buf230 = buf228[1]
        assert_size_stride(buf230, (256, 1024, 1, 1), (1024, 1, 1024, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf230, 16, 'torch.ops.aten.convolution_backward.default')
        del buf228
        buf232 = reinterpret_tensor(buf224, (1024, 128), (1, 1024), 0); del buf224  # reuse
        buf234 = reinterpret_tensor(buf222, (1024, 128), (1, 1024), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_87], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_17.run(relu_27, buf201, buf229, convolution_30, unsqueeze_478, buf232, buf234, 131072, 784, stream=stream0)
        buf233 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(buf232, buf233, 1024, 128, stream=stream0)
        buf235 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf237 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_87], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19.run(buf234, squeeze_91, buf235, buf237, 1024, 128, stream=stream0)
        buf238 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_87], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_20.run(relu_27, buf201, buf229, convolution_30, unsqueeze_478, buf235, squeeze_91, buf233, primals_186, buf238, 102760448, stream=stream0)
        del convolution_30
        del primals_186
        del squeeze_91
        del unsqueeze_478
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        buf239 = torch.ops.aten.convolution_backward.default(buf238, relu_26, convert_element_type_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf238
        del convert_element_type_91
        buf240 = buf239[0]
        assert_size_stride(buf240, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf240, 16, 'torch.ops.aten.convolution_backward.default')
        buf241 = buf239[1]
        assert_size_stride(buf241, (1024, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf241, 16, 'torch.ops.aten.convolution_backward.default')
        del buf239
        buf243 = reinterpret_tensor(buf234, (256, 512), (1, 256), 0); del buf234  # reuse
        buf245 = reinterpret_tensor(buf232, (256, 512), (1, 256), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_84], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_26, buf240, convolution_29, unsqueeze_490, buf243, buf245, 131072, 196, stream=stream0)
        buf244 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf243, buf244, 256, 512, stream=stream0)
        buf246 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf247 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_84], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf245, squeeze_88, buf246, buf247, 256, 512, stream=stream0)
        buf248 = relu_26; del relu_26  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_84], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf248, buf240, convolution_29, unsqueeze_490, buf246, squeeze_88, buf244, primals_180, 25690112, stream=stream0)
        del buf240
        del convolution_29
        del primals_180
        del squeeze_88
        del unsqueeze_490
        # Topologically Sorted Source Nodes: [scalar_tensor, out_84], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf249 = torch.ops.aten.convolution_backward.default(buf248, relu_25, convert_element_type_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf248
        del convert_element_type_88
        buf250 = buf249[0]
        assert_size_stride(buf250, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf250, 16, 'torch.ops.aten.convolution_backward.default')
        buf251 = buf249[1]
        assert_size_stride(buf251, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf251, 16, 'torch.ops.aten.convolution_backward.default')
        del buf249
        buf253 = buf245; del buf245  # reuse
        buf255 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_81], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_25, buf250, convolution_28, unsqueeze_502, buf253, buf255, 131072, 196, stream=stream0)
        buf254 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf253, buf254, 256, 512, stream=stream0)
        buf256 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf257 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_81], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf255, squeeze_85, buf256, buf257, 256, 512, stream=stream0)
        buf258 = relu_25; del relu_25  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_81], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf258, buf250, convolution_28, unsqueeze_502, buf256, squeeze_85, buf254, primals_174, 25690112, stream=stream0)
        del buf250
        del convolution_28
        del primals_174
        del squeeze_85
        del unsqueeze_502
        # Topologically Sorted Source Nodes: [scalar_tensor, out_81], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf259 = torch.ops.aten.convolution_backward.default(buf258, relu_24, convert_element_type_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf258
        del convert_element_type_85
        buf260 = buf259[0]
        assert_size_stride(buf260, (512, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf260, 16, 'torch.ops.aten.convolution_backward.default')
        buf261 = buf259[1]
        assert_size_stride(buf261, (256, 1024, 1, 1), (1024, 1, 1024, 1024), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf261, 16, 'torch.ops.aten.convolution_backward.default')
        del buf259
        buf263 = empty_strided_cuda((512, 1024, 14, 14), (200704, 1, 14336, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_28.run(relu_24, relu_27, buf201, buf229, buf260, buf263, 102760448, stream=stream0)
        del buf201
        del buf229
        del buf260
        del relu_24
        del relu_27
        buf264 = reinterpret_tensor(buf255, (1024, 128), (1, 1024), 0); del buf255  # reuse
        buf266 = reinterpret_tensor(buf253, (1024, 128), (1, 1024), 0); del buf253  # reuse
        buf274 = reinterpret_tensor(buf78, (1024, 128), (1, 1024), 0); del buf78  # reuse
        buf276 = reinterpret_tensor(buf68, (1024, 128), (1, 1024), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [input_6, out_77], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_29.run(buf263, convolution_27, unsqueeze_514, convolution_26, unsqueeze_526, buf264, buf266, buf274, buf276, 131072, 784, stream=stream0)
        buf265 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(buf264, buf265, 1024, 128, stream=stream0)
        del buf264
        buf275 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_18.run(buf274, buf275, 1024, 128, stream=stream0)
        del buf274
        buf267 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf268 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19.run(buf266, squeeze_82, buf267, buf268, 1024, 128, stream=stream0)
        buf277 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        buf278 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_19.run(buf276, squeeze_79, buf277, buf278, 1024, 128, stream=stream0)
        buf269 = convolution_27; del convolution_27  # reuse
        buf279 = convolution_26; del convolution_26  # reuse
        # Topologically Sorted Source Nodes: [input_6, out_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_30.run(buf269, buf279, buf263, unsqueeze_514, buf267, squeeze_82, buf265, primals_168, unsqueeze_526, buf277, squeeze_79, buf275, primals_162, 102760448, stream=stream0)
        del buf263
        del buf267
        del buf277
        del primals_162
        del primals_168
        del squeeze_79
        del squeeze_82
        del unsqueeze_514
        del unsqueeze_526
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        buf270 = torch.ops.aten.convolution_backward.default(buf269, relu_21, convert_element_type_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf269
        del convert_element_type_82
        buf271 = buf270[0]
        assert_size_stride(buf271, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf271, 16, 'torch.ops.aten.convolution_backward.default')
        buf272 = buf270[1]
        assert_size_stride(buf272, (1024, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf272, 16, 'torch.ops.aten.convolution_backward.default')
        del buf270
        # Topologically Sorted Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        buf280 = torch.ops.aten.convolution_backward.default(buf279, relu_23, convert_element_type_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf279
        del convert_element_type_79
        buf281 = buf280[0]
        assert_size_stride(buf281, (512, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf281, 16, 'torch.ops.aten.convolution_backward.default')
        buf282 = buf280[1]
        assert_size_stride(buf282, (1024, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf282, 16, 'torch.ops.aten.convolution_backward.default')
        del buf280
        buf284 = reinterpret_tensor(buf276, (256, 512), (1, 256), 0); del buf276  # reuse
        buf286 = reinterpret_tensor(buf266, (256, 512), (1, 256), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_74], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_21.run(relu_23, buf281, convolution_25, unsqueeze_538, buf284, buf286, 131072, 196, stream=stream0)
        buf285 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf284, buf285, 256, 512, stream=stream0)
        buf287 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf288 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_74], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf286, squeeze_76, buf287, buf288, 256, 512, stream=stream0)
        buf289 = relu_23; del relu_23  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_74], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_24.run(buf289, buf281, convolution_25, unsqueeze_538, buf287, squeeze_76, buf285, primals_156, 25690112, stream=stream0)
        del buf281
        del convolution_25
        del primals_156
        del squeeze_76
        del unsqueeze_538
        # Topologically Sorted Source Nodes: [scalar_tensor, out_74], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf290 = torch.ops.aten.convolution_backward.default(buf289, relu_22, convert_element_type_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf289
        del convert_element_type_76
        buf291 = buf290[0]
        assert_size_stride(buf291, (512, 256, 28, 28), (200704, 1, 7168, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf291, 16, 'torch.ops.aten.convolution_backward.default')
        buf292 = buf290[1]
        assert_size_stride(buf292, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf292, 16, 'torch.ops.aten.convolution_backward.default')
        del buf290
        buf294 = buf286; del buf286  # reuse
        buf296 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_71], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_31.run(relu_22, buf291, convolution_24, unsqueeze_550, buf294, buf296, 131072, 784, stream=stream0)
        buf295 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_22.run(buf294, buf295, 256, 512, stream=stream0)
        buf297 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf298 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_71], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_23.run(buf296, squeeze_73, buf297, buf298, 256, 512, stream=stream0)
        buf299 = relu_22; del relu_22  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_71], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_32.run(buf299, buf291, convolution_24, unsqueeze_550, buf297, squeeze_73, buf295, primals_150, 102760448, stream=stream0)
        del buf291
        del convolution_24
        del primals_150
        del squeeze_73
        del unsqueeze_550
        # Topologically Sorted Source Nodes: [scalar_tensor, out_71], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf300 = torch.ops.aten.convolution_backward.default(buf299, relu_21, convert_element_type_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf299
        del convert_element_type_73
        buf301 = buf300[0]
        assert_size_stride(buf301, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf301, 16, 'torch.ops.aten.convolution_backward.default')
        buf302 = buf300[1]
        assert_size_stride(buf302, (256, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf302, 16, 'torch.ops.aten.convolution_backward.default')
        del buf300
        buf304 = buf90; del buf90  # reuse
        buf306 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_67], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_33.run(relu_21, buf271, buf301, convolution_23, unsqueeze_562, buf304, buf306, 100352, 2048, stream=stream0)
        buf305 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf304, buf305, 512, 196, stream=stream0)
        buf307 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf309 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_67], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf306, squeeze_70, buf307, buf309, 512, 196, stream=stream0)
        buf310 = empty_strided_cuda((512, 512, 28, 28), (401408, 1, 14336, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_67], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_34.run(relu_21, buf271, buf301, convolution_23, unsqueeze_562, buf307, squeeze_70, buf305, primals_144, buf310, 205520896, stream=stream0)
        del convolution_23
        del primals_144
        del squeeze_70
        del unsqueeze_562
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        buf311 = torch.ops.aten.convolution_backward.default(buf310, relu_20, convert_element_type_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf310
        del convert_element_type_70
        buf312 = buf311[0]
        assert_size_stride(buf312, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf312, 16, 'torch.ops.aten.convolution_backward.default')
        buf313 = buf311[1]
        assert_size_stride(buf313, (512, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf313, 16, 'torch.ops.aten.convolution_backward.default')
        del buf311
        buf315 = reinterpret_tensor(buf296, (128, 1024), (1, 128), 0); del buf296  # reuse
        buf317 = reinterpret_tensor(buf294, (128, 1024), (1, 128), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_64], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35.run(relu_20, buf312, convolution_22, unsqueeze_574, buf315, buf317, 131072, 392, stream=stream0)
        buf316 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf315, buf316, 128, 1024, stream=stream0)
        buf318 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf319 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_64], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37.run(buf317, squeeze_67, buf318, buf319, 128, 1024, stream=stream0)
        buf320 = relu_20; del relu_20  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_64], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf320, buf312, convolution_22, unsqueeze_574, buf318, squeeze_67, buf316, primals_138, 51380224, stream=stream0)
        del buf312
        del convolution_22
        del primals_138
        del squeeze_67
        del unsqueeze_574
        # Topologically Sorted Source Nodes: [scalar_tensor, out_64], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf321 = torch.ops.aten.convolution_backward.default(buf320, relu_19, convert_element_type_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf320
        del convert_element_type_67
        buf322 = buf321[0]
        assert_size_stride(buf322, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf322, 16, 'torch.ops.aten.convolution_backward.default')
        buf323 = buf321[1]
        assert_size_stride(buf323, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf323, 16, 'torch.ops.aten.convolution_backward.default')
        del buf321
        buf325 = buf317; del buf317  # reuse
        buf327 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_61], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35.run(relu_19, buf322, convolution_21, unsqueeze_586, buf325, buf327, 131072, 392, stream=stream0)
        buf326 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf325, buf326, 128, 1024, stream=stream0)
        buf328 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf329 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_61], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37.run(buf327, squeeze_64, buf328, buf329, 128, 1024, stream=stream0)
        buf330 = relu_19; del relu_19  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_61], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf330, buf322, convolution_21, unsqueeze_586, buf328, squeeze_64, buf326, primals_132, 51380224, stream=stream0)
        del buf322
        del convolution_21
        del primals_132
        del squeeze_64
        del unsqueeze_586
        # Topologically Sorted Source Nodes: [scalar_tensor, out_61], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf331 = torch.ops.aten.convolution_backward.default(buf330, relu_18, convert_element_type_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf330
        del convert_element_type_64
        buf332 = buf331[0]
        assert_size_stride(buf332, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf332, 16, 'torch.ops.aten.convolution_backward.default')
        buf333 = buf331[1]
        assert_size_stride(buf333, (128, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf333, 16, 'torch.ops.aten.convolution_backward.default')
        del buf331
        buf335 = relu_18; del relu_18  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_threshold_backward_39.run(buf335, relu_21, buf271, buf301, buf332, 205520896, stream=stream0)
        del buf271
        del buf301
        del buf332
        del relu_21
        buf336 = buf306; del buf306  # reuse
        buf338 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_40.run(buf335, convolution_20, unsqueeze_598, buf336, buf338, 100352, 2048, stream=stream0)
        buf337 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf336, buf337, 512, 196, stream=stream0)
        buf339 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf340 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf338, squeeze_61, buf339, buf340, 512, 196, stream=stream0)
        buf341 = convolution_20; del convolution_20  # reuse
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_41.run(buf341, buf335, unsqueeze_598, buf339, squeeze_61, buf337, primals_126, 205520896, stream=stream0)
        del primals_126
        del squeeze_61
        del unsqueeze_598
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf342 = torch.ops.aten.convolution_backward.default(buf341, relu_17, convert_element_type_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_61
        buf343 = buf342[0]
        assert_size_stride(buf343, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf343, 16, 'torch.ops.aten.convolution_backward.default')
        buf344 = buf342[1]
        assert_size_stride(buf344, (512, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf344, 16, 'torch.ops.aten.convolution_backward.default')
        del buf342
        buf346 = buf327; del buf327  # reuse
        buf348 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_54], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35.run(relu_17, buf343, convolution_19, unsqueeze_610, buf346, buf348, 131072, 392, stream=stream0)
        buf347 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf346, buf347, 128, 1024, stream=stream0)
        buf349 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf350 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_54], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37.run(buf348, squeeze_58, buf349, buf350, 128, 1024, stream=stream0)
        buf351 = relu_17; del relu_17  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_54], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf351, buf343, convolution_19, unsqueeze_610, buf349, squeeze_58, buf347, primals_120, 51380224, stream=stream0)
        del buf343
        del convolution_19
        del primals_120
        del squeeze_58
        del unsqueeze_610
        # Topologically Sorted Source Nodes: [scalar_tensor, out_54], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf352 = torch.ops.aten.convolution_backward.default(buf351, relu_16, convert_element_type_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf351
        del convert_element_type_58
        buf353 = buf352[0]
        assert_size_stride(buf353, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf353, 16, 'torch.ops.aten.convolution_backward.default')
        buf354 = buf352[1]
        assert_size_stride(buf354, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf354, 16, 'torch.ops.aten.convolution_backward.default')
        del buf352
        buf356 = buf348; del buf348  # reuse
        buf358 = buf346; del buf346  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_51], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35.run(relu_16, buf353, convolution_18, unsqueeze_622, buf356, buf358, 131072, 392, stream=stream0)
        buf357 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf356, buf357, 128, 1024, stream=stream0)
        buf359 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf360 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_51], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37.run(buf358, squeeze_55, buf359, buf360, 128, 1024, stream=stream0)
        buf361 = relu_16; del relu_16  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_51], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf361, buf353, convolution_18, unsqueeze_622, buf359, squeeze_55, buf357, primals_114, 51380224, stream=stream0)
        del buf353
        del convolution_18
        del primals_114
        del squeeze_55
        del unsqueeze_622
        # Topologically Sorted Source Nodes: [scalar_tensor, out_51], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf362 = torch.ops.aten.convolution_backward.default(buf361, relu_15, convert_element_type_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf361
        del convert_element_type_55
        buf363 = buf362[0]
        assert_size_stride(buf363, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf363, 16, 'torch.ops.aten.convolution_backward.default')
        buf364 = buf362[1]
        assert_size_stride(buf364, (128, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf364, 16, 'torch.ops.aten.convolution_backward.default')
        del buf362
        buf366 = buf338; del buf338  # reuse
        buf368 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_47], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_33.run(relu_15, buf335, buf363, convolution_17, unsqueeze_634, buf366, buf368, 100352, 2048, stream=stream0)
        buf367 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf366, buf367, 512, 196, stream=stream0)
        buf369 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf371 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_47], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf368, squeeze_52, buf369, buf371, 512, 196, stream=stream0)
        buf372 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_47], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_34.run(relu_15, buf335, buf363, convolution_17, unsqueeze_634, buf369, squeeze_52, buf367, primals_108, buf372, 205520896, stream=stream0)
        del convolution_17
        del primals_108
        del squeeze_52
        del unsqueeze_634
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        buf373 = torch.ops.aten.convolution_backward.default(buf372, relu_14, convert_element_type_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf372
        del convert_element_type_52
        buf374 = buf373[0]
        assert_size_stride(buf374, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf374, 16, 'torch.ops.aten.convolution_backward.default')
        buf375 = buf373[1]
        assert_size_stride(buf375, (512, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf375, 16, 'torch.ops.aten.convolution_backward.default')
        del buf373
        buf377 = buf358; del buf358  # reuse
        buf379 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_44], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35.run(relu_14, buf374, convolution_16, unsqueeze_646, buf377, buf379, 131072, 392, stream=stream0)
        buf378 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf377, buf378, 128, 1024, stream=stream0)
        buf380 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf381 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_44], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37.run(buf379, squeeze_49, buf380, buf381, 128, 1024, stream=stream0)
        buf382 = relu_14; del relu_14  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_44], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf382, buf374, convolution_16, unsqueeze_646, buf380, squeeze_49, buf378, primals_102, 51380224, stream=stream0)
        del buf374
        del convolution_16
        del primals_102
        del squeeze_49
        del unsqueeze_646
        # Topologically Sorted Source Nodes: [scalar_tensor, out_44], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf383 = torch.ops.aten.convolution_backward.default(buf382, relu_13, convert_element_type_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf382
        del convert_element_type_49
        buf384 = buf383[0]
        assert_size_stride(buf384, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf384, 16, 'torch.ops.aten.convolution_backward.default')
        buf385 = buf383[1]
        assert_size_stride(buf385, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf385, 16, 'torch.ops.aten.convolution_backward.default')
        del buf383
        buf387 = buf379; del buf379  # reuse
        buf389 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_41], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35.run(relu_13, buf384, convolution_15, unsqueeze_658, buf387, buf389, 131072, 392, stream=stream0)
        buf388 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf387, buf388, 128, 1024, stream=stream0)
        buf390 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf391 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_41], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37.run(buf389, squeeze_46, buf390, buf391, 128, 1024, stream=stream0)
        buf392 = relu_13; del relu_13  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_41], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf392, buf384, convolution_15, unsqueeze_658, buf390, squeeze_46, buf388, primals_96, 51380224, stream=stream0)
        del buf384
        del convolution_15
        del primals_96
        del squeeze_46
        del unsqueeze_658
        # Topologically Sorted Source Nodes: [scalar_tensor, out_41], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf393 = torch.ops.aten.convolution_backward.default(buf392, relu_12, convert_element_type_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf392
        del convert_element_type_46
        buf394 = buf393[0]
        assert_size_stride(buf394, (512, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf394, 16, 'torch.ops.aten.convolution_backward.default')
        buf395 = buf393[1]
        assert_size_stride(buf395, (128, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf395, 16, 'torch.ops.aten.convolution_backward.default')
        del buf393
        buf397 = empty_strided_cuda((512, 512, 28, 28), (401408, 1, 14336, 512), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_42.run(relu_12, relu_15, buf335, buf363, buf394, buf397, 205520896, stream=stream0)
        del buf335
        del buf363
        del buf394
        del relu_12
        del relu_15
        buf398 = buf368; del buf368  # reuse
        buf400 = buf366; del buf366  # reuse
        buf408 = empty_strided_cuda((512, 196), (1, 512), torch.float32)
        buf410 = empty_strided_cuda((512, 196), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, out_37], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_43.run(buf397, convolution_14, unsqueeze_670, convolution_13, unsqueeze_682, buf398, buf400, buf408, buf410, 100352, 2048, stream=stream0)
        buf399 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf398, buf399, 512, 196, stream=stream0)
        del buf398
        buf409 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_5.run(buf408, buf409, 512, 196, stream=stream0)
        del buf408
        buf401 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf402 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf400, squeeze_43, buf401, buf402, 512, 196, stream=stream0)
        buf411 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf412 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_6.run(buf410, squeeze_40, buf411, buf412, 512, 196, stream=stream0)
        buf403 = convolution_14; del convolution_14  # reuse
        buf413 = convolution_13; del convolution_13  # reuse
        # Topologically Sorted Source Nodes: [input_4, out_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_44.run(buf403, buf413, buf397, unsqueeze_670, buf401, squeeze_43, buf399, primals_90, unsqueeze_682, buf411, squeeze_40, buf409, primals_84, 205520896, stream=stream0)
        del buf397
        del buf401
        del buf411
        del primals_84
        del primals_90
        del squeeze_40
        del squeeze_43
        del unsqueeze_670
        del unsqueeze_682
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        buf404 = torch.ops.aten.convolution_backward.default(buf403, relu_9, convert_element_type_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf403
        del convert_element_type_43
        buf405 = buf404[0]
        assert_size_stride(buf405, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf405, 16, 'torch.ops.aten.convolution_backward.default')
        buf406 = buf404[1]
        assert_size_stride(buf406, (512, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf406, 16, 'torch.ops.aten.convolution_backward.default')
        del buf404
        # Topologically Sorted Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        buf414 = torch.ops.aten.convolution_backward.default(buf413, relu_11, convert_element_type_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf413
        del convert_element_type_40
        buf415 = buf414[0]
        assert_size_stride(buf415, (512, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf415, 16, 'torch.ops.aten.convolution_backward.default')
        buf416 = buf414[1]
        assert_size_stride(buf416, (512, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf416, 16, 'torch.ops.aten.convolution_backward.default')
        del buf414
        buf418 = buf389; del buf389  # reuse
        buf420 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_34], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_35.run(relu_11, buf415, convolution_12, unsqueeze_694, buf418, buf420, 131072, 392, stream=stream0)
        buf419 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_36.run(buf418, buf419, 128, 1024, stream=stream0)
        buf421 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf422 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_34], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_37.run(buf420, squeeze_37, buf421, buf422, 128, 1024, stream=stream0)
        buf423 = relu_11; del relu_11  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_34], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_38.run(buf423, buf415, convolution_12, unsqueeze_694, buf421, squeeze_37, buf419, primals_78, 51380224, stream=stream0)
        del buf415
        del convolution_12
        del primals_78
        del squeeze_37
        del unsqueeze_694
        # Topologically Sorted Source Nodes: [scalar_tensor, out_34], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf424 = torch.ops.aten.convolution_backward.default(buf423, relu_10, convert_element_type_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf423
        del convert_element_type_37
        buf425 = buf424[0]
        assert_size_stride(buf425, (512, 128, 56, 56), (401408, 1, 7168, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf425, 16, 'torch.ops.aten.convolution_backward.default')
        buf426 = buf424[1]
        assert_size_stride(buf426, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf426, 16, 'torch.ops.aten.convolution_backward.default')
        del buf424
        buf428 = reinterpret_tensor(buf410, (128, 784), (1, 128), 0); del buf410  # reuse
        buf430 = reinterpret_tensor(buf400, (128, 784), (1, 128), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_45.run(relu_10, buf425, convolution_11, unsqueeze_706, buf428, buf430, 100352, 2048, stream=stream0)
        buf429 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_46.run(buf428, buf429, 128, 784, stream=stream0)
        del buf428
        buf431 = empty_strided_cuda((128, ), (1, ), torch.float32)
        buf432 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_47.run(buf430, squeeze_34, buf431, buf432, 128, 784, stream=stream0)
        del buf430
        buf433 = relu_10; del relu_10  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_48.run(buf433, buf425, convolution_11, unsqueeze_706, buf431, squeeze_34, buf429, primals_72, 205520896, stream=stream0)
        del buf425
        del buf431
        del convolution_11
        del primals_72
        del squeeze_34
        del unsqueeze_706
        # Topologically Sorted Source Nodes: [scalar_tensor, out_31], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf434 = torch.ops.aten.convolution_backward.default(buf433, relu_9, convert_element_type_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf433
        del convert_element_type_34
        buf435 = buf434[0]
        assert_size_stride(buf435, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf435, 16, 'torch.ops.aten.convolution_backward.default')
        buf436 = buf434[1]
        assert_size_stride(buf436, (128, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf436, 16, 'torch.ops.aten.convolution_backward.default')
        del buf434
        buf438 = empty_strided_cuda((256, 784), (1, 256), torch.float32)
        buf440 = empty_strided_cuda((256, 784), (1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_27], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_49.run(relu_9, buf405, buf435, convolution_10, unsqueeze_718, buf438, buf440, 200704, 2048, stream=stream0)
        buf439 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf438, buf439, 256, 784, stream=stream0)
        buf441 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf443 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_27], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_51.run(buf440, squeeze_31, buf441, buf443, 256, 784, stream=stream0)
        buf444 = empty_strided_cuda((512, 256, 56, 56), (802816, 1, 14336, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_27], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_52.run(relu_9, buf405, buf435, convolution_10, unsqueeze_718, buf441, squeeze_31, buf439, primals_66, buf444, 411041792, stream=stream0)
        del convolution_10
        del primals_66
        del squeeze_31
        del unsqueeze_718
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        buf445 = torch.ops.aten.convolution_backward.default(buf444, relu_8, convert_element_type_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf444
        del convert_element_type_31
        buf446 = buf445[0]
        assert_size_stride(buf446, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf446, 16, 'torch.ops.aten.convolution_backward.default')
        buf447 = buf445[1]
        assert_size_stride(buf447, (256, 64, 1, 1), (64, 1, 64, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf447, 16, 'torch.ops.aten.convolution_backward.default')
        del buf445
        buf449 = empty_strided_cuda((64, 1024), (1, 64), torch.float32)
        buf451 = empty_strided_cuda((64, 1024), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_24], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53.run(relu_8, buf446, convolution_9, unsqueeze_730, buf449, buf451, 65536, 1568, stream=stream0)
        buf450 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(buf449, buf450, 64, 1024, stream=stream0)
        buf452 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf453 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_24], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55.run(buf451, squeeze_28, buf452, buf453, 64, 1024, stream=stream0)
        buf454 = relu_8; del relu_8  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_24], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf454, buf446, convolution_9, unsqueeze_730, buf452, squeeze_28, buf450, primals_60, 102760448, stream=stream0)
        del buf446
        del convolution_9
        del primals_60
        del squeeze_28
        del unsqueeze_730
        # Topologically Sorted Source Nodes: [scalar_tensor, out_24], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf455 = torch.ops.aten.convolution_backward.default(buf454, relu_7, convert_element_type_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf454
        del convert_element_type_28
        buf456 = buf455[0]
        assert_size_stride(buf456, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf456, 16, 'torch.ops.aten.convolution_backward.default')
        buf457 = buf455[1]
        assert_size_stride(buf457, (64, 64, 3, 3), (576, 1, 192, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf457, 16, 'torch.ops.aten.convolution_backward.default')
        del buf455
        buf459 = buf451; del buf451  # reuse
        buf461 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_21], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53.run(relu_7, buf456, convolution_8, unsqueeze_742, buf459, buf461, 65536, 1568, stream=stream0)
        buf460 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(buf459, buf460, 64, 1024, stream=stream0)
        buf462 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf463 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_21], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55.run(buf461, squeeze_25, buf462, buf463, 64, 1024, stream=stream0)
        buf464 = relu_7; del relu_7  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_21], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf464, buf456, convolution_8, unsqueeze_742, buf462, squeeze_25, buf460, primals_54, 102760448, stream=stream0)
        del buf456
        del convolution_8
        del primals_54
        del squeeze_25
        del unsqueeze_742
        # Topologically Sorted Source Nodes: [scalar_tensor, out_21], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf465 = torch.ops.aten.convolution_backward.default(buf464, relu_6, convert_element_type_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf464
        del convert_element_type_25
        buf466 = buf465[0]
        assert_size_stride(buf466, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf466, 16, 'torch.ops.aten.convolution_backward.default')
        buf467 = buf465[1]
        assert_size_stride(buf467, (64, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf467, 16, 'torch.ops.aten.convolution_backward.default')
        del buf465
        buf469 = relu_6; del relu_6  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_threshold_backward_57.run(buf469, relu_9, buf405, buf435, buf466, 411041792, stream=stream0)
        del buf405
        del buf435
        del relu_9
        buf470 = buf440; del buf440  # reuse
        buf472 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_58.run(buf469, convolution_7, unsqueeze_754, buf470, buf472, 200704, 2048, stream=stream0)
        buf471 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf470, buf471, 256, 784, stream=stream0)
        buf473 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf474 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_51.run(buf472, squeeze_22, buf473, buf474, 256, 784, stream=stream0)
        buf475 = convolution_7; del convolution_7  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_59.run(buf475, buf469, unsqueeze_754, buf473, squeeze_22, buf471, primals_48, 411041792, stream=stream0)
        del primals_48
        del squeeze_22
        del unsqueeze_754
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf476 = torch.ops.aten.convolution_backward.default(buf475, relu_5, convert_element_type_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_22
        buf477 = buf476[0]
        assert_size_stride(buf477, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf477, 16, 'torch.ops.aten.convolution_backward.default')
        buf478 = buf476[1]
        assert_size_stride(buf478, (256, 64, 1, 1), (64, 1, 64, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf478, 16, 'torch.ops.aten.convolution_backward.default')
        del buf476
        buf480 = buf461; del buf461  # reuse
        buf482 = buf459; del buf459  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_14], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53.run(relu_5, buf477, convolution_6, unsqueeze_766, buf480, buf482, 65536, 1568, stream=stream0)
        buf481 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(buf480, buf481, 64, 1024, stream=stream0)
        buf483 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf484 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_14], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55.run(buf482, squeeze_19, buf483, buf484, 64, 1024, stream=stream0)
        buf485 = relu_5; del relu_5  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_14], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf485, buf477, convolution_6, unsqueeze_766, buf483, squeeze_19, buf481, primals_42, 102760448, stream=stream0)
        del buf477
        del convolution_6
        del primals_42
        del squeeze_19
        del unsqueeze_766
        # Topologically Sorted Source Nodes: [scalar_tensor, out_14], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf486 = torch.ops.aten.convolution_backward.default(buf485, relu_4, convert_element_type_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf485
        del convert_element_type_19
        buf487 = buf486[0]
        assert_size_stride(buf487, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf487, 16, 'torch.ops.aten.convolution_backward.default')
        buf488 = buf486[1]
        assert_size_stride(buf488, (64, 64, 3, 3), (576, 1, 192, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf488, 16, 'torch.ops.aten.convolution_backward.default')
        del buf486
        buf490 = buf482; del buf482  # reuse
        buf492 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_11], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53.run(relu_4, buf487, convolution_5, unsqueeze_778, buf490, buf492, 65536, 1568, stream=stream0)
        buf491 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(buf490, buf491, 64, 1024, stream=stream0)
        buf493 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf494 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_11], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55.run(buf492, squeeze_16, buf493, buf494, 64, 1024, stream=stream0)
        buf495 = relu_4; del relu_4  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_11], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf495, buf487, convolution_5, unsqueeze_778, buf493, squeeze_16, buf491, primals_36, 102760448, stream=stream0)
        del buf487
        del convolution_5
        del primals_36
        del squeeze_16
        del unsqueeze_778
        # Topologically Sorted Source Nodes: [scalar_tensor, out_11], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf496 = torch.ops.aten.convolution_backward.default(buf495, relu_3, convert_element_type_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf495
        del convert_element_type_16
        buf497 = buf496[0]
        assert_size_stride(buf497, (512, 256, 56, 56), (802816, 1, 14336, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf497, 16, 'torch.ops.aten.convolution_backward.default')
        buf498 = buf496[1]
        assert_size_stride(buf498, (64, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf498, 16, 'torch.ops.aten.convolution_backward.default')
        del buf496
        buf500 = buf472; del buf472  # reuse
        buf502 = buf470; del buf470  # reuse
        buf511 = empty_strided_cuda((256, 784), (1, 256), torch.float32)
        buf513 = empty_strided_cuda((256, 784), (1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, input_2, out_7], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_60.run(relu_3, buf469, buf497, convolution_4, unsqueeze_790, convolution_3, unsqueeze_802, buf500, buf502, buf511, buf513, 200704, 2048, stream=stream0)
        buf501 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf500, buf501, 256, 784, stream=stream0)
        del buf500
        buf512 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_50.run(buf511, buf512, 256, 784, stream=stream0)
        del buf511
        buf503 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf505 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, input_2], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_51.run(buf502, squeeze_13, buf503, buf505, 256, 784, stream=stream0)
        buf514 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf516 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_7], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_51.run(buf513, squeeze_10, buf514, buf516, 256, 784, stream=stream0)
        buf506 = buf475; del buf475  # reuse
        buf517 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, input_2, out_7], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_61.run(relu_3, buf469, buf497, convolution_4, unsqueeze_790, buf503, squeeze_13, buf501, primals_30, convolution_3, unsqueeze_802, buf514, squeeze_10, buf512, primals_24, buf506, buf517, 411041792, stream=stream0)
        del buf469
        del buf497
        del buf503
        del buf514
        del convolution_3
        del convolution_4
        del primals_24
        del primals_30
        del relu_3
        del squeeze_10
        del squeeze_13
        del unsqueeze_790
        del unsqueeze_802
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        buf507 = torch.ops.aten.convolution_backward.default(buf506, getitem_2, convert_element_type_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf506
        del convert_element_type_13
        buf508 = buf507[0]
        assert_size_stride(buf508, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf508, 16, 'torch.ops.aten.convolution_backward.default')
        buf509 = buf507[1]
        assert_size_stride(buf509, (256, 64, 1, 1), (64, 1, 64, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf509, 16, 'torch.ops.aten.convolution_backward.default')
        del buf507
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        buf518 = torch.ops.aten.convolution_backward.default(buf517, relu_2, convert_element_type_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del convert_element_type_10
        buf519 = buf518[0]
        assert_size_stride(buf519, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf519, 16, 'torch.ops.aten.convolution_backward.default')
        buf520 = buf518[1]
        assert_size_stride(buf520, (256, 64, 1, 1), (64, 1, 64, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf520, 16, 'torch.ops.aten.convolution_backward.default')
        del buf518
        buf522 = buf492; del buf492  # reuse
        buf524 = buf490; del buf490  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_4], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53.run(relu_2, buf519, convolution_2, unsqueeze_814, buf522, buf524, 65536, 1568, stream=stream0)
        buf523 = buf493; del buf493  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(buf522, buf523, 64, 1024, stream=stream0)
        buf525 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf526 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_4], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55.run(buf524, squeeze_7, buf525, buf526, 64, 1024, stream=stream0)
        buf527 = relu_2; del relu_2  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_4], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf527, buf519, convolution_2, unsqueeze_814, buf525, squeeze_7, buf523, primals_18, 102760448, stream=stream0)
        del buf519
        del convolution_2
        del primals_18
        del squeeze_7
        del unsqueeze_814
        # Topologically Sorted Source Nodes: [scalar_tensor, out_4], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf528 = torch.ops.aten.convolution_backward.default(buf527, relu_1, convert_element_type_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf527
        del convert_element_type_7
        buf529 = buf528[0]
        assert_size_stride(buf529, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf529, 16, 'torch.ops.aten.convolution_backward.default')
        buf530 = buf528[1]
        assert_size_stride(buf530, (64, 64, 3, 3), (576, 1, 192, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf530, 16, 'torch.ops.aten.convolution_backward.default')
        del buf528
        buf532 = buf524; del buf524  # reuse
        buf534 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_53.run(relu_1, buf529, convolution_1, unsqueeze_826, buf532, buf534, 65536, 1568, stream=stream0)
        buf533 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_54.run(buf532, buf533, 64, 1024, stream=stream0)
        buf535 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf536 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, out_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_55.run(buf534, squeeze_4, buf535, buf536, 64, 1024, stream=stream0)
        buf537 = relu_1; del relu_1  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, out_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_56.run(buf537, buf529, convolution_1, unsqueeze_826, buf535, squeeze_4, buf533, primals_12, 102760448, stream=stream0)
        del buf529
        del convolution_1
        del primals_12
        del squeeze_4
        del unsqueeze_826
        # Topologically Sorted Source Nodes: [scalar_tensor, out_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf538 = torch.ops.aten.convolution_backward.default(buf537, getitem_2, convert_element_type_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf537
        del convert_element_type_4
        del getitem_2
        buf539 = buf538[0]
        assert_size_stride(buf539, (512, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf539, 16, 'torch.ops.aten.convolution_backward.default')
        buf540 = buf538[1]
        assert_size_stride(buf540, (64, 64, 1, 1), (64, 1, 64, 64), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf540, 16, 'torch.ops.aten.convolution_backward.default')
        del buf538
        buf542 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_62.run(buf542, buf539, 102760448, stream=stream0)
        del buf539
        buf543 = reinterpret_tensor(buf517, (512, 64, 112, 112), (802816, 1, 7168, 64), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_63.run(getitem_3, buf542, buf543, 411041792, stream=stream0)
        del buf542
        del getitem_3
        buf544 = reinterpret_tensor(buf513, (64, 3136), (1, 64), 0); del buf513  # reuse
        buf546 = reinterpret_tensor(buf502, (64, 3136), (1, 64), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, x_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_64.run(relu, buf543, convolution, unsqueeze_838, buf544, buf546, 200704, 2048, stream=stream0)
        buf545 = buf535; del buf535  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_batch_norm_backward_threshold_backward_65.run(buf544, buf545, 64, 3136, stream=stream0)
        del buf544
        buf547 = empty_strided_cuda((64, ), (1, ), torch.float32)
        buf548 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [scalar_tensor, x_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_native_batch_norm_backward_threshold_backward_66.run(buf546, squeeze_1, buf547, buf548, 64, 3136, stream=stream0)
        del buf546
        buf549 = relu; del relu  # reuse
        # Topologically Sorted Source Nodes: [scalar_tensor, x_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_67.run(buf549, buf543, convolution, unsqueeze_838, buf547, squeeze_1, buf545, primals_6, 411041792, stream=stream0)
        del buf543
        del buf547
        del convolution
        del primals_6
        del squeeze_1
        del unsqueeze_838
        # Topologically Sorted Source Nodes: [scalar_tensor, x_1], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten._native_batch_norm_legit_functional, aten.convolution_backward]
        buf550 = torch.ops.aten.convolution_backward.default(buf549, convert_element_type_1, convert_element_type, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf549
        del convert_element_type
        del convert_element_type_1
        buf551 = buf550[1]
        assert_size_stride(buf551, (64, 3, 7, 7), (147, 1, 21, 3), 'torch.ops.aten.convolution_backward.default')
        assert_alignment(buf551, 16, 'torch.ops.aten.convolution_backward.default')
        del buf550
        buf3 = empty_strided_cuda((1, 100, 4), (400, 1, 100), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_68.run(tangents_1, buf3, 400, 128, stream=stream0)
        buf4 = empty_strided_cuda((1, 100), (100, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_69.run(buf3, buf4, 100, 4, stream=stream0)
        del buf3
        buf541 = empty_strided_cuda((64, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_70.run(buf540, buf541, 4096, stream=stream0)
        del buf540
        buf552 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_71.run(buf551, buf552, 9408, stream=stream0)
        del buf551
        buf448 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(buf447, buf448, 16384, stream=stream0)
        del buf447
        buf468 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(buf467, buf468, 16384, stream=stream0)
        del buf467
        buf479 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(buf478, buf479, 16384, stream=stream0)
        del buf478
        buf499 = empty_strided_cuda((64, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(buf498, buf499, 16384, stream=stream0)
        del buf498
        buf510 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(buf509, buf510, 16384, stream=stream0)
        del buf509
        buf521 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(buf520, buf521, 16384, stream=stream0)
        del buf520
        buf437 = empty_strided_cuda((128, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_73.run(buf436, buf437, 32768, stream=stream0)
        del buf436
        buf1 = empty_strided_cuda((104, 512), (1, 104), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_74.run(tangents_1, buf1, 53248, stream=stream0)
        del tangents_1
        buf458 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf457, buf458, 36864, stream=stream0)
        del buf457
        buf489 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf488, buf489, 36864, stream=stream0)
        del buf488
        buf531 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_75.run(buf530, buf531, 36864, stream=stream0)
        del buf530
        buf314 = reinterpret_tensor(buf534, (512, 128, 1, 1), (128, 1, 128, 128), 0); del buf534  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_76.run(buf313, buf314, 65536, stream=stream0)
        del buf313
        buf334 = reinterpret_tensor(buf532, (128, 512, 1, 1), (512, 1, 512, 512), 0); del buf532  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_76.run(buf333, buf334, 65536, stream=stream0)
        del buf333
        buf345 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_76.run(buf344, buf345, 65536, stream=stream0)
        del buf344
        buf365 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_76.run(buf364, buf365, 65536, stream=stream0)
        del buf364
        buf376 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_76.run(buf375, buf376, 65536, stream=stream0)
        del buf375
        buf396 = empty_strided_cuda((128, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_76.run(buf395, buf396, 65536, stream=stream0)
        del buf395
        buf417 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_76.run(buf416, buf417, 65536, stream=stream0)
        del buf416
        buf303 = reinterpret_tensor(buf420, (256, 512, 1, 1), (512, 1, 512, 512), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_77.run(buf302, buf303, 131072, stream=stream0)
        del buf302
        buf407 = reinterpret_tensor(buf418, (512, 256, 1, 1), (256, 1, 256, 256), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_77.run(buf406, buf407, 131072, stream=stream0)
        del buf406
        buf324 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_78.run(buf323, buf324, 147456, stream=stream0)
        del buf323
        buf355 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_78.run(buf354, buf355, 147456, stream=stream0)
        del buf354
        buf386 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_78.run(buf385, buf386, 147456, stream=stream0)
        del buf385
        buf427 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_78.run(buf426, buf427, 147456, stream=stream0)
        del buf426
        buf2 = empty_strided_cuda((104, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1, view, out=buf2)
        del buf1
        del view
        buf5 = empty_strided_cuda((100, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_79.run(buf2, buf5, 204800, stream=stream0)
        del buf2
        buf118 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf117, buf118, 262144, stream=stream0)
        del buf117
        buf138 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf137, buf138, 262144, stream=stream0)
        del buf137
        buf149 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf148, buf149, 262144, stream=stream0)
        del buf148
        buf169 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf168, buf169, 262144, stream=stream0)
        del buf168
        buf180 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf179, buf180, 262144, stream=stream0)
        del buf179
        buf200 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf199, buf200, 262144, stream=stream0)
        del buf199
        buf211 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf210, buf211, 262144, stream=stream0)
        del buf210
        buf231 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf230, buf231, 262144, stream=stream0)
        del buf230
        buf242 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf241, buf242, 262144, stream=stream0)
        del buf241
        buf262 = empty_strided_cuda((256, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf261, buf262, 262144, stream=stream0)
        del buf261
        buf283 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_80.run(buf282, buf283, 262144, stream=stream0)
        del buf282
        buf107 = empty_strided_cuda((512, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(buf106, buf107, 524288, stream=stream0)
        del buf106
        buf273 = empty_strided_cuda((1024, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_81.run(buf272, buf273, 524288, stream=stream0)
        del buf272
        buf128 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_82.run(buf127, buf128, 589824, stream=stream0)
        del buf127
        buf159 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_82.run(buf158, buf159, 589824, stream=stream0)
        del buf158
        buf190 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_82.run(buf189, buf190, 589824, stream=stream0)
        del buf189
        buf221 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_82.run(buf220, buf221, 589824, stream=stream0)
        del buf220
        buf252 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_82.run(buf251, buf252, 589824, stream=stream0)
        del buf251
        buf293 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_82.run(buf292, buf293, 589824, stream=stream0)
        del buf292
        buf15 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_83.run(buf14, buf15, 1048576, stream=stream0)
        del buf14
        buf35 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_83.run(buf34, buf35, 1048576, stream=stream0)
        del buf34
        buf46 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_83.run(buf45, buf46, 1048576, stream=stream0)
        del buf45
        buf66 = empty_strided_cuda((512, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_83.run(buf65, buf66, 1048576, stream=stream0)
        del buf65
        buf87 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_83.run(buf86, buf87, 1048576, stream=stream0)
        del buf86
        buf77 = empty_strided_cuda((2048, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_84.run(buf76, buf77, 2097152, stream=stream0)
        del buf76
        buf25 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_85.run(buf24, buf25, 2359296, stream=stream0)
        del buf24
        buf56 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_85.run(buf55, buf56, 2359296, stream=stream0)
        del buf55
        buf97 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_85.run(buf96, buf97, 2359296, stream=stream0)
        del buf96
    return (buf552, None, None, None, None, buf548, buf545, buf541, None, None, None, buf536, buf533, buf531, None, None, None, buf526, buf523, buf521, None, None, None, buf516, buf512, buf510, None, None, None, buf505, buf501, buf499, None, None, None, buf494, buf491, buf489, None, None, None, buf484, buf481, buf479, None, None, None, buf474, buf471, buf468, None, None, None, buf463, buf460, buf458, None, None, None, buf453, buf450, buf448, None, None, None, buf443, buf439, buf437, None, None, None, buf432, buf429, buf427, None, None, None, buf422, buf419, buf417, None, None, None, buf412, buf409, buf407, None, None, None, buf402, buf399, buf396, None, None, None, buf391, buf388, buf386, None, None, None, buf381, buf378, buf376, None, None, None, buf371, buf367, buf365, None, None, None, buf360, buf357, buf355, None, None, None, buf350, buf347, buf345, None, None, None, buf340, buf337, buf334, None, None, None, buf329, buf326, buf324, None, None, None, buf319, buf316, buf314, None, None, None, buf309, buf305, buf303, None, None, None, buf298, buf295, buf293, None, None, None, buf288, buf285, buf283, None, None, None, buf278, buf275, buf273, None, None, None, buf268, buf265, buf262, None, None, None, buf257, buf254, buf252, None, None, None, buf247, buf244, buf242, None, None, None, buf237, buf233, buf231, None, None, None, buf226, buf223, buf221, None, None, None, buf216, buf213, buf211, None, None, None, buf206, buf203, buf200, None, None, None, buf195, buf192, buf190, None, None, None, buf185, buf182, buf180, None, None, None, buf175, buf171, buf169, None, None, None, buf164, buf161, buf159, None, None, None, buf154, buf151, buf149, None, None, None, buf144, buf141, buf138, None, None, None, buf133, buf130, buf128, None, None, None, buf123, buf120, buf118, None, None, None, buf113, buf109, buf107, None, None, None, buf102, buf99, buf97, None, None, None, buf92, buf89, buf87, None, None, None, buf82, buf79, buf77, None, None, None, buf72, buf69, buf66, None, None, None, buf61, buf58, buf56, None, None, None, buf51, buf48, buf46, None, None, None, buf41, buf37, buf35, None, None, None, buf30, buf27, buf25, None, None, None, buf20, buf17, buf15, None, None, None, buf10, buf7, buf5, reinterpret_tensor(buf4, (100, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_1 = rand_strided((512, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.bfloat16)
    convolution = rand_strided((512, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.bfloat16)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((512, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_2 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    getitem_3 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.int8)
    convert_element_type_4 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    convolution_1 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_7 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    convolution_2 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_10 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    convolution_3 = rand_strided((512, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_13 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    convolution_4 = rand_strided((512, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((512, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_16 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_5 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    squeeze_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_19 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    convolution_6 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    squeeze_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_22 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    convolution_7 = rand_strided((512, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((512, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_25 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_8 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_28 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.bfloat16)
    convolution_9 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    squeeze_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((512, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_31 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.bfloat16)
    convolution_10 = rand_strided((512, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((512, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_34 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_11 = rand_strided((512, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((512, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_37 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_12 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_40 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_13 = rand_strided((512, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_43 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_14 = rand_strided((512, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((512, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_46 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_15 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_49 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_16 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_52 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_17 = rand_strided((512, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((512, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_55 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_18 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_58 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_19 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_61 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_20 = rand_strided((512, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_61 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((512, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_64 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_21 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_67 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_22 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    squeeze_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((512, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_70 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.bfloat16)
    convolution_23 = rand_strided((512, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_21 = rand_strided((512, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_73 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_24 = rand_strided((512, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((512, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_76 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_25 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_23 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_79 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_26 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    squeeze_79 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_82 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_27 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    squeeze_82 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_24 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_85 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convolution_28 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_25 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_88 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_29 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_26 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_91 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_30 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    squeeze_91 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_27 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_94 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convolution_31 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_28 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_97 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_32 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_29 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_100 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_33 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    squeeze_100 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_30 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_103 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convolution_34 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_31 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_106 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_35 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_32 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_109 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_36 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    squeeze_109 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_33 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_112 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convolution_37 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_34 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_115 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_38 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_35 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_118 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_39 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    squeeze_118 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_36 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_121 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convolution_40 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_37 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_124 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_41 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    squeeze_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_38 = rand_strided((512, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_127 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.bfloat16)
    convolution_42 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    squeeze_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_39 = rand_strided((512, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_130 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convolution_43 = rand_strided((512, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_40 = rand_strided((512, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_133 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_44 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_41 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_136 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_45 = rand_strided((512, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bfloat16)
    squeeze_136 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    convert_element_type_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.bfloat16)
    convolution_46 = rand_strided((512, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bfloat16)
    squeeze_139 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_42 = rand_strided((512, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_142 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.bfloat16)
    convolution_47 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_43 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_145 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_48 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_44 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_148 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_49 = rand_strided((512, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bfloat16)
    squeeze_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_45 = rand_strided((512, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_151 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.bfloat16)
    convolution_50 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_46 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_154 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_51 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    squeeze_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_47 = rand_strided((512, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_157 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.bfloat16)
    convolution_52 = rand_strided((512, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bfloat16)
    squeeze_157 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1 = rand_strided((100, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    le = rand_strided((512, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.bool)
    unsqueeze_214 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_706 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_754 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_802 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((512, 100), (100, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, primals_132, primals_138, primals_144, primals_150, primals_156, primals_162, primals_168, primals_174, primals_180, primals_186, primals_192, primals_198, primals_204, primals_210, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, primals_252, primals_258, primals_264, primals_270, primals_276, primals_282, primals_288, primals_294, primals_300, primals_306, primals_312, primals_318, convert_element_type, convert_element_type_1, convolution, squeeze_1, relu, getitem_2, getitem_3, convert_element_type_4, convolution_1, squeeze_4, relu_1, convert_element_type_7, convolution_2, squeeze_7, relu_2, convert_element_type_10, convolution_3, squeeze_10, convert_element_type_13, convolution_4, squeeze_13, relu_3, convert_element_type_16, convolution_5, squeeze_16, relu_4, convert_element_type_19, convolution_6, squeeze_19, relu_5, convert_element_type_22, convolution_7, squeeze_22, relu_6, convert_element_type_25, convolution_8, squeeze_25, relu_7, convert_element_type_28, convolution_9, squeeze_28, relu_8, convert_element_type_31, convolution_10, squeeze_31, relu_9, convert_element_type_34, convolution_11, squeeze_34, relu_10, convert_element_type_37, convolution_12, squeeze_37, relu_11, convert_element_type_40, convolution_13, squeeze_40, convert_element_type_43, convolution_14, squeeze_43, relu_12, convert_element_type_46, convolution_15, squeeze_46, relu_13, convert_element_type_49, convolution_16, squeeze_49, relu_14, convert_element_type_52, convolution_17, squeeze_52, relu_15, convert_element_type_55, convolution_18, squeeze_55, relu_16, convert_element_type_58, convolution_19, squeeze_58, relu_17, convert_element_type_61, convolution_20, squeeze_61, relu_18, convert_element_type_64, convolution_21, squeeze_64, relu_19, convert_element_type_67, convolution_22, squeeze_67, relu_20, convert_element_type_70, convolution_23, squeeze_70, relu_21, convert_element_type_73, convolution_24, squeeze_73, relu_22, convert_element_type_76, convolution_25, squeeze_76, relu_23, convert_element_type_79, convolution_26, squeeze_79, convert_element_type_82, convolution_27, squeeze_82, relu_24, convert_element_type_85, convolution_28, squeeze_85, relu_25, convert_element_type_88, convolution_29, squeeze_88, relu_26, convert_element_type_91, convolution_30, squeeze_91, relu_27, convert_element_type_94, convolution_31, squeeze_94, relu_28, convert_element_type_97, convolution_32, squeeze_97, relu_29, convert_element_type_100, convolution_33, squeeze_100, relu_30, convert_element_type_103, convolution_34, squeeze_103, relu_31, convert_element_type_106, convolution_35, squeeze_106, relu_32, convert_element_type_109, convolution_36, squeeze_109, relu_33, convert_element_type_112, convolution_37, squeeze_112, relu_34, convert_element_type_115, convolution_38, squeeze_115, relu_35, convert_element_type_118, convolution_39, squeeze_118, relu_36, convert_element_type_121, convolution_40, squeeze_121, relu_37, convert_element_type_124, convolution_41, squeeze_124, relu_38, convert_element_type_127, convolution_42, squeeze_127, relu_39, convert_element_type_130, convolution_43, squeeze_130, relu_40, convert_element_type_133, convolution_44, squeeze_133, relu_41, convert_element_type_136, convolution_45, squeeze_136, convert_element_type_139, convolution_46, squeeze_139, relu_42, convert_element_type_142, convolution_47, squeeze_142, relu_43, convert_element_type_145, convolution_48, squeeze_145, relu_44, convert_element_type_148, convolution_49, squeeze_148, relu_45, convert_element_type_151, convolution_50, squeeze_151, relu_46, convert_element_type_154, convolution_51, squeeze_154, relu_47, convert_element_type_157, convolution_52, squeeze_157, view, permute_1, le, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
