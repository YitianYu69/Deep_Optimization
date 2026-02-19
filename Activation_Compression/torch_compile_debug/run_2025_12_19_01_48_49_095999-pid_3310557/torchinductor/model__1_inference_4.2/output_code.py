# AOT ID: ['1_inference']
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


# kernel path: /tmp/torchinductor_yyu496/kb/ckbz32exozdsnc2cn7of32r3up6rpd737rfj72byn7ch6egqepnk.py
# Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   conv1 => convert_element_type, convert_element_type_1, convolution
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_1, %convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__to_copy_convolution_0 = async_compile.triton('triton_poi_fused__to_copy_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/7n/c7nwi6pslu6ujvwa6tco7udmxe4vgfgckmm44uufbtffvowxr5ul.py
# Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   conv1 => convert_element_type, convert_element_type_1, convolution
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.bfloat16), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_1, %convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__to_copy_convolution_1 = async_compile.triton('triton_poi_fused__to_copy_convolution_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 75264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/66/c66e5zwc7jqwafqp5cz72evtp6bpfyngtg5lbaekogvrf2gdjhr3.py
# Topologically Sorted Source Nodes: [bn1, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   bn1 => add_11, convert_element_type_4, mul_8, mul_9, sub_2
#   relu => relu
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_3), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %unsqueeze_5), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %unsqueeze_7), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.bfloat16), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_4,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 12544) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fg/cfg6ht75q2crqpixxoxxmbfqgxnseuztjo2k3kgw3kklk7xjjhqy.py
# Topologically Sorted Source Nodes: [bn1, relu, maxpool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   bn1 => add_11, convert_element_type_4, mul_8, mul_9, sub_2
#   maxpool => _low_memory_max_pool_with_offsets
#   relu => relu
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_3), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %unsqueeze_5), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %unsqueeze_7), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.bfloat16), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_4,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 56) % 56)
    x0 = (xindex % 56)
    x3 = xindex // 56
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + 2*x0 + 224*x3), tmp10, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-112) + 2*x0 + 224*x3), tmp16, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-111) + 2*x0 + 224*x3), tmp23, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 224*x3), tmp30, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 224*x3), tmp33, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 224*x3), tmp36, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (111 + 2*x0 + 224*x3), tmp43, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (112 + 2*x0 + 224*x3), tmp46, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (113 + 2*x0 + 224*x3), tmp49, eviction_policy='evict_last', other=float("-inf")).to(tl.float32)
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x4), tmp51, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/2c/c2cqx5t7bllwvudbadksgv7qh6togrnw35zlgdo6okcl72fexgzg.py
# Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_conv1 => convert_element_type_5, convolution_1
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg7_1, torch.bfloat16), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__to_copy_convolution_4 = async_compile.triton('triton_poi_fused__to_copy_convolution_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/el/celjy2qdgz36kq7po6mswovgrfveywhmoyzztblruenny6et65ia.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_relu, layer1_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_bn1 => add_38, convert_element_type_8, mul_24, mul_25, sub_8
#   layer1_0_conv2 => convert_element_type_9, convolution_2
#   layer1_0_relu => relu_1
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_11), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %unsqueeze_13), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %unsqueeze_15), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_38, torch.bfloat16), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_8,), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.bfloat16), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %convert_element_type_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 3136) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/oq/coqjmn2jono7qjvxhbze7uwam3f5shzaz4ypykwkcfyyeueg5y5z.py
# Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_relu, layer1_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_bn1 => add_38, convert_element_type_8, mul_24, mul_25, sub_8
#   layer1_0_conv2 => convert_element_type_9, convolution_2
#   layer1_0_relu => relu_1
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_11), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %unsqueeze_13), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %unsqueeze_15), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_38, torch.bfloat16), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_8,), kwargs = {})
#   %convert_element_type_9 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg12_1, torch.bfloat16), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %convert_element_type_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/lk/clkdn6d6mlxxqjb4fagz37bxh2r2tm6x62wa3hwms6gfjlvlym4x.py
# Topologically Sorted Source Nodes: [layer1_0_bn2, layer1_0_relu_1, layer1_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer1_0_bn2 => add_55, convert_element_type_12, mul_36, mul_37, sub_12
#   layer1_0_conv3 => convert_element_type_13, convolution_3
#   layer1_0_relu_1 => relu_2
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_19), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %unsqueeze_21), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %unsqueeze_23), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_55, torch.bfloat16), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_12,), kwargs = {})
#   %convert_element_type_13 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.bfloat16), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %convert_element_type_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fa/cfa26waxiasemsl52rifzmjyz47slpmyzqbimommv36vh6zezyqu.py
# Topologically Sorted Source Nodes: [layer1_0_bn3, layer1_0_downsample_1, add, layer1_0_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add => add_90
#   layer1_0_bn3 => add_72, convert_element_type_16, mul_48, mul_49, sub_16
#   layer1_0_downsample_1 => add_84, convert_element_type_20, mul_58, mul_59, sub_19
#   layer1_0_relu_2 => relu_3
# Graph fragment:
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_27), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %unsqueeze_29), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_31), kwargs = {})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_72, torch.bfloat16), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_35), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_37), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_39), kwargs = {})
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_84, torch.bfloat16), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_16, %convert_element_type_20), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_90,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 3136) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp20 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp19 - tmp20
    tmp23 = tmp22 + tmp5
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = (tmp8 / tmp24)
    tmp26 = tmp25 * tmp10
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp17 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/tq/ctql47ksterrmeavktfpdougdrl6pdeahnuctzbgegbzivwfsffk.py
# Topologically Sorted Source Nodes: [layer1_1_bn3, add_1, layer1_1_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_1 => add_147
#   layer1_1_bn3 => add_141, convert_element_type_32, mul_96, mul_97, sub_32
#   layer1_1_relu_2 => relu_6
# Graph fragment:
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_59), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %unsqueeze_61), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %unsqueeze_63), kwargs = {})
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_141, torch.bfloat16), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_32, %relu_3), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_147,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 3136) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/72/c72viqqnblgsfkpfuo3642aizicfv6hqn7ywctolelmdxwjjtm3d.py
# Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_conv1 => convert_element_type_45, convolution_11
# Graph fragment:
#   %convert_element_type_45 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg57_1, torch.bfloat16), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %convert_element_type_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__to_copy_convolution_10 = async_compile.triton('triton_poi_fused__to_copy_convolution_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/c6/cc6ly66yqqsium5ok4mixyubskru25nfhddtg5o2dujmnppb454h.py
# Topologically Sorted Source Nodes: [layer2_0_bn1, layer2_0_relu, layer2_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_bn1 => add_221, convert_element_type_48, mul_148, mul_149, sub_50
#   layer2_0_conv2 => convert_element_type_49, convolution_12
#   layer2_0_relu => relu_10
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_91), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_93), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_95), kwargs = {})
#   %convert_element_type_48 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_221, torch.bfloat16), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_48,), kwargs = {})
#   %convert_element_type_49 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg62_1, torch.bfloat16), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %convert_element_type_49, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 3136) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ye/cye7vr7vqkzuu3f5l4hzrmzhkm4jni5ja6brtwfjxw34eufph4y4.py
# Topologically Sorted Source Nodes: [layer2_0_bn1, layer2_0_relu, layer2_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_bn1 => add_221, convert_element_type_48, mul_148, mul_149, sub_50
#   layer2_0_conv2 => convert_element_type_49, convolution_12
#   layer2_0_relu => relu_10
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_91), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_93), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_95), kwargs = {})
#   %convert_element_type_48 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_221, torch.bfloat16), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_48,), kwargs = {})
#   %convert_element_type_49 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg62_1, torch.bfloat16), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %convert_element_type_49, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/vd/cvd4dz6bal76x4p6qyyuysyo5ezonbixwrvxvio2vdmpdmi5lfzu.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_relu_1, layer2_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_bn2 => add_238, convert_element_type_52, mul_160, mul_161, sub_54
#   layer2_0_conv3 => convert_element_type_53, convolution_13
#   layer2_0_relu_1 => relu_11
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_99), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_101), kwargs = {})
#   %add_238 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_103), kwargs = {})
#   %convert_element_type_52 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_238, torch.bfloat16), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_52,), kwargs = {})
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg67_1, torch.bfloat16), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 784) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gg/cggjqyig4mqh5c6ryhofxxmjlz4xtl3fstjrff4ado2kkk6c3l5v.py
# Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_relu_1, layer2_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_bn2 => add_238, convert_element_type_52, mul_160, mul_161, sub_54
#   layer2_0_conv3 => convert_element_type_53, convolution_13
#   layer2_0_relu_1 => relu_11
# Graph fragment:
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_54, %unsqueeze_99), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_160, %unsqueeze_101), kwargs = {})
#   %add_238 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_161, %unsqueeze_103), kwargs = {})
#   %convert_element_type_52 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_238, torch.bfloat16), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_52,), kwargs = {})
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg67_1, torch.bfloat16), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fs/cfsr3fzetmfcwxwzh4wmqlt7n3r2mcaiavhlhbi53aksygo67sil.py
# Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer2_0_downsample_0 => convert_element_type_57, convolution_14
# Graph fragment:
#   %convert_element_type_57 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg72_1, torch.bfloat16), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %convert_element_type_57, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__to_copy_convolution_15 = async_compile.triton('triton_poi_fused__to_copy_convolution_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1048576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/w3/cw3ouk3miqvupsdsb6ec5iimhuaeht5egkcgigbv4raacnfsy2js.py
# Topologically Sorted Source Nodes: [layer2_0_bn3, layer2_0_downsample_1, add_3, layer2_0_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_3 => add_273
#   layer2_0_bn3 => add_255, convert_element_type_56, mul_172, mul_173, sub_58
#   layer2_0_downsample_1 => add_267, convert_element_type_60, mul_182, mul_183, sub_61
#   layer2_0_relu_2 => relu_12
# Graph fragment:
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_107), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %unsqueeze_109), kwargs = {})
#   %add_255 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_173, %unsqueeze_111), kwargs = {})
#   %convert_element_type_56 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_255, torch.bfloat16), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_115), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_182, %unsqueeze_117), kwargs = {})
#   %add_267 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_183, %unsqueeze_119), kwargs = {})
#   %convert_element_type_60 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_267, torch.bfloat16), kwargs = {})
#   %add_273 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_56, %convert_element_type_60), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_273,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 784) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp20 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp19 - tmp20
    tmp23 = tmp22 + tmp5
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = (tmp8 / tmp24)
    tmp26 = tmp25 * tmp10
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp17 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/w4/cw4nms3ijpirdoqn73mex2aat4i5cbwu66ll2bnt4wptuz3ozy3p.py
# Topologically Sorted Source Nodes: [layer2_1_bn3, add_4, layer2_1_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_4 => add_330
#   layer2_1_bn3 => add_324, convert_element_type_72, mul_220, mul_221, sub_74
#   layer2_1_relu_2 => relu_15
# Graph fragment:
#   %sub_74 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_74, %unsqueeze_139), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_220, %unsqueeze_141), kwargs = {})
#   %add_324 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_221, %unsqueeze_143), kwargs = {})
#   %convert_element_type_72 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_324, torch.bfloat16), kwargs = {})
#   %add_330 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_72, %relu_12), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_330,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 784) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/zn/czn733yiu2om5cz7wye4rtq4s4ox7bs3v5phzo73u57xc4k5j3pf.py
# Topologically Sorted Source Nodes: [layer3_0_bn1, layer3_0_relu, layer3_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_bn1 => add_461, convert_element_type_100, mul_310, mul_311, sub_105
#   layer3_0_conv2 => convert_element_type_101, convolution_25
#   layer3_0_relu => relu_22
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_195), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_197), kwargs = {})
#   %add_461 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_199), kwargs = {})
#   %convert_element_type_100 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_461, torch.bfloat16), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_100,), kwargs = {})
#   %convert_element_type_101 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg127_1, torch.bfloat16), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %convert_element_type_101, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 784) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ot/cotxckxpck3r3lazy4uomz2qxz74q755vzzvjrvqjcu7i4cwztiz.py
# Topologically Sorted Source Nodes: [layer3_0_bn1, layer3_0_relu, layer3_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_bn1 => add_461, convert_element_type_100, mul_310, mul_311, sub_105
#   layer3_0_conv2 => convert_element_type_101, convolution_25
#   layer3_0_relu => relu_22
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_195), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_197), kwargs = {})
#   %add_461 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_199), kwargs = {})
#   %convert_element_type_100 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_461, torch.bfloat16), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_100,), kwargs = {})
#   %convert_element_type_101 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg127_1, torch.bfloat16), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %convert_element_type_101, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/5n/c5niogw2bbzs5u3gkmmjeeijkwjxo2z6ssqi7yl54r7a6w6dhnua.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_relu_1, layer3_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_bn2 => add_478, convert_element_type_104, mul_322, mul_323, sub_109
#   layer3_0_conv3 => convert_element_type_105, convolution_26
#   layer3_0_relu_1 => relu_23
# Graph fragment:
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %unsqueeze_203), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_322, %unsqueeze_205), kwargs = {})
#   %add_478 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %unsqueeze_207), kwargs = {})
#   %convert_element_type_104 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_478, torch.bfloat16), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_104,), kwargs = {})
#   %convert_element_type_105 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg132_1, torch.bfloat16), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %convert_element_type_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 196) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/jc/cjctmd7lrinjlgtx266sddbul6xc377ctsjnktp3dim5b7zzqrjg.py
# Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_relu_1, layer3_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_bn2 => add_478, convert_element_type_104, mul_322, mul_323, sub_109
#   layer3_0_conv3 => convert_element_type_105, convolution_26
#   layer3_0_relu_1 => relu_23
# Graph fragment:
#   %sub_109 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_109, %unsqueeze_203), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_322, %unsqueeze_205), kwargs = {})
#   %add_478 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %unsqueeze_207), kwargs = {})
#   %convert_element_type_104 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_478, torch.bfloat16), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_104,), kwargs = {})
#   %convert_element_type_105 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg132_1, torch.bfloat16), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %convert_element_type_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2097152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/u2/cu2rrdkarlrc7pzqps2fpfw55cj5aceyymh4opom3gsj26ttdfta.py
# Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer3_0_downsample_0 => convert_element_type_109, convolution_27
# Graph fragment:
#   %convert_element_type_109 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg137_1, torch.bfloat16), kwargs = {})
#   %convolution_27 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %convert_element_type_109, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__to_copy_convolution_22 = async_compile.triton('triton_poi_fused__to_copy_convolution_22', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4194304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/6g/c6gx723hvs2arnjika42ll2wylqxurpwsa5c6mmhc4ds5bwohgw6.py
# Topologically Sorted Source Nodes: [layer3_0_bn3, layer3_0_downsample_1, add_7, layer3_0_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_7 => add_513
#   layer3_0_bn3 => add_495, convert_element_type_108, mul_334, mul_335, sub_113
#   layer3_0_downsample_1 => add_507, convert_element_type_112, mul_344, mul_345, sub_116
#   layer3_0_relu_2 => relu_24
# Graph fragment:
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %unsqueeze_211), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %unsqueeze_213), kwargs = {})
#   %add_495 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %unsqueeze_215), kwargs = {})
#   %convert_element_type_108 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_495, torch.bfloat16), kwargs = {})
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_116, %unsqueeze_219), kwargs = {})
#   %mul_345 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_344, %unsqueeze_221), kwargs = {})
#   %add_507 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_345, %unsqueeze_223), kwargs = {})
#   %convert_element_type_112 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_507, torch.bfloat16), kwargs = {})
#   %add_513 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_108, %convert_element_type_112), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_513,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 196) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp20 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp19 - tmp20
    tmp23 = tmp22 + tmp5
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = (tmp8 / tmp24)
    tmp26 = tmp25 * tmp10
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp17 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/fo/cfoyta4zyqm6eqqs5urjdbz2j2vongpuzmqp66z42feq7qio5s35.py
# Topologically Sorted Source Nodes: [layer3_1_bn3, add_8, layer3_1_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_8 => add_570
#   layer3_1_bn3 => add_564, convert_element_type_124, mul_382, mul_383, sub_129
#   layer3_1_relu_2 => relu_27
# Graph fragment:
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_243), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_382, %unsqueeze_245), kwargs = {})
#   %add_564 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %unsqueeze_247), kwargs = {})
#   %convert_element_type_124 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_564, torch.bfloat16), kwargs = {})
#   %add_570 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_124, %relu_24), kwargs = {})
#   %relu_27 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_570,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 196) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/iu/ciumltv7zr442yt375e3vibaogq4wn24b6lic25z7bkemm6n76vc.py
# Topologically Sorted Source Nodes: [layer4_0_bn1, layer4_0_relu, layer4_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_bn1 => add_815, convert_element_type_176, mul_548, mul_549, sub_186
#   layer4_0_conv2 => convert_element_type_177, convolution_44
#   layer4_0_relu => relu_40
# Graph fragment:
#   %sub_186 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_548 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_186, %unsqueeze_347), kwargs = {})
#   %mul_549 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_548, %unsqueeze_349), kwargs = {})
#   %add_815 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_549, %unsqueeze_351), kwargs = {})
#   %convert_element_type_176 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_815, torch.bfloat16), kwargs = {})
#   %relu_40 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_176,), kwargs = {})
#   %convert_element_type_177 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg222_1, torch.bfloat16), kwargs = {})
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_40, %convert_element_type_177, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 196) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/to/ctoj2ujxj5jefpm73xh2mk6wz4hie7hlly6rlwfuzyjmtsxph6gv.py
# Topologically Sorted Source Nodes: [layer4_0_bn1, layer4_0_relu, layer4_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_bn1 => add_815, convert_element_type_176, mul_548, mul_549, sub_186
#   layer4_0_conv2 => convert_element_type_177, convolution_44
#   layer4_0_relu => relu_40
# Graph fragment:
#   %sub_186 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_548 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_186, %unsqueeze_347), kwargs = {})
#   %mul_549 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_548, %unsqueeze_349), kwargs = {})
#   %add_815 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_549, %unsqueeze_351), kwargs = {})
#   %convert_element_type_176 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_815, torch.bfloat16), kwargs = {})
#   %relu_40 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_176,), kwargs = {})
#   %convert_element_type_177 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg222_1, torch.bfloat16), kwargs = {})
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_40, %convert_element_type_177, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ka/ckaj5qtp3nl6dv5magegzbnh2lleniz6e7xa5b7a5ovx2cehq22h.py
# Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_relu_1, layer4_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_bn2 => add_832, convert_element_type_180, mul_560, mul_561, sub_190
#   layer4_0_conv3 => convert_element_type_181, convolution_45
#   layer4_0_relu_1 => relu_41
# Graph fragment:
#   %sub_190 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_560 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_190, %unsqueeze_355), kwargs = {})
#   %mul_561 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_560, %unsqueeze_357), kwargs = {})
#   %add_832 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_561, %unsqueeze_359), kwargs = {})
#   %convert_element_type_180 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_832, torch.bfloat16), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_180,), kwargs = {})
#   %convert_element_type_181 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg227_1, torch.bfloat16), kwargs = {})
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %convert_element_type_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 49) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ur/curqab5nk23zzp3yp53pf5d227bz6awrgerngkxzcxezyqtztevq.py
# Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_relu_1, layer4_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_bn2 => add_832, convert_element_type_180, mul_560, mul_561, sub_190
#   layer4_0_conv3 => convert_element_type_181, convolution_45
#   layer4_0_relu_1 => relu_41
# Graph fragment:
#   %sub_190 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_560 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_190, %unsqueeze_355), kwargs = {})
#   %mul_561 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_560, %unsqueeze_357), kwargs = {})
#   %add_832 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_561, %unsqueeze_359), kwargs = {})
#   %convert_element_type_180 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_832, torch.bfloat16), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convert_element_type_180,), kwargs = {})
#   %convert_element_type_181 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg227_1, torch.bfloat16), kwargs = {})
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %convert_element_type_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_28', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/sh/csh6tb6k52ifpod4bvbnqtcwyw2k6xciduxznq6bvexxlrzwgfy6.py
# Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
# Source node to ATen node mapping:
#   layer4_0_downsample_0 => convert_element_type_185, convolution_46
# Graph fragment:
#   %convert_element_type_185 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg232_1, torch.bfloat16), kwargs = {})
#   %convolution_46 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_39, %convert_element_type_185, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__to_copy_convolution_29 = async_compile.triton('triton_poi_fused__to_copy_convolution_29', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 16777216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_convolution_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/ro/croogis5aowxka3b6qf2elz35s55ubsykqsmndzmkbmdhqu2dkwi.py
# Topologically Sorted Source Nodes: [layer4_0_bn3, layer4_0_downsample_1, add_13, layer4_0_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_13 => add_867
#   layer4_0_bn3 => add_849, convert_element_type_184, mul_572, mul_573, sub_194
#   layer4_0_downsample_1 => add_861, convert_element_type_188, mul_582, mul_583, sub_197
#   layer4_0_relu_2 => relu_42
# Graph fragment:
#   %sub_194 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_572 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_194, %unsqueeze_363), kwargs = {})
#   %mul_573 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_572, %unsqueeze_365), kwargs = {})
#   %add_849 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_573, %unsqueeze_367), kwargs = {})
#   %convert_element_type_184 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_849, torch.bfloat16), kwargs = {})
#   %sub_197 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_369), kwargs = {})
#   %mul_582 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_197, %unsqueeze_371), kwargs = {})
#   %mul_583 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_582, %unsqueeze_373), kwargs = {})
#   %add_861 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_583, %unsqueeze_375), kwargs = {})
#   %convert_element_type_188 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_861, torch.bfloat16), kwargs = {})
#   %add_867 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_184, %convert_element_type_188), kwargs = {})
#   %relu_42 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_867,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 49) % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), xmask).to(tl.float32)
    tmp20 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp19 - tmp20
    tmp23 = tmp22 + tmp5
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = (tmp8 / tmp24)
    tmp26 = tmp25 * tmp10
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp17 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(in_out_ptr0 + (x3), tmp35, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/gg/cggqwdb22w4jays2bqa55n5qihpgpethr4mn2i7d5bdmlzw5zrpr.py
# Topologically Sorted Source Nodes: [layer4_1_bn3, add_14, layer4_1_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   add_14 => add_924
#   layer4_1_bn3 => add_918, convert_element_type_200, mul_620, mul_621, sub_210
#   layer4_1_relu_2 => relu_45
# Graph fragment:
#   %sub_210 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_393), kwargs = {})
#   %mul_620 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_210, %unsqueeze_395), kwargs = {})
#   %mul_621 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_620, %unsqueeze_397), kwargs = {})
#   %add_918 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_621, %unsqueeze_399), kwargs = {})
#   %convert_element_type_200 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_918, torch.bfloat16), kwargs = {})
#   %add_924 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_200, %relu_42), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_924,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 49) % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x3), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/uv/cuvandcn3yr53zoxr4qomvyznt6ynhezteicuei4dxkme53kkbue.py
# Topologically Sorted Source Nodes: [layer4_2_bn3, add_15, layer4_2_relu_2, avgpool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   add_15 => add_981
#   avgpool => mean
#   layer4_2_bn3 => add_975, convert_element_type_212, mul_658, mul_659, sub_223
#   layer4_2_relu_2 => relu_48
# Graph fragment:
#   %sub_223 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_417), kwargs = {})
#   %mul_658 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_223, %unsqueeze_419), kwargs = {})
#   %mul_659 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_658, %unsqueeze_421), kwargs = {})
#   %add_975 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_659, %unsqueeze_423), kwargs = {})
#   %convert_element_type_212 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_975, torch.bfloat16), kwargs = {})
#   %add_981 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_212, %relu_45), kwargs = {})
#   %relu_48 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_981,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_48, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_32 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 49*x3), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (r0_2 + 49*x3), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = (tmp8 / tmp7)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1, 1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp25 = tl.where(r0_mask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 49.0
    tmp28 = (tmp26 / tmp27)
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/t5/ct5ezxh5dwlnlvhaouefzlh2w42zayxw6gj3f6xwell5jsdcxgtk.py
# Topologically Sorted Source Nodes: [fc], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   fc => convert_element_type_214
# Graph fragment:
#   %convert_element_type_214 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg267_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_33 = async_compile.triton('triton_poi_fused__to_copy_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1638400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_yyu496/6u/c6u6l4hbemfmrcht3zisor3jmqdqcjzpxxtokrfvd26l32ooxd7g.py
# Topologically Sorted Source Nodes: [fc], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   fc => convert_element_type_213
# Graph fragment:
#   %convert_element_type_213 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg268_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_34 = async_compile.triton('triton_poi_fused__to_copy_34', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'B2183E6715F23468D0C05FDB4E88B7BA0C7A9BC4BB41D90BFD2CEB4FE6D5FDEF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1 = args
    args.clear()
    s77 = arg1_1
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg2_1, (s77, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (64, ), (1, ))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (64, ), (1, ))
    assert_size_stride(arg37_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (64, ), (1, ))
    assert_size_stride(arg47_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (64, ), (1, ))
    assert_size_stride(arg52_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (128, ), (1, ))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (128, ), (1, ))
    assert_size_stride(arg97_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, ), (1, ))
    assert_size_stride(arg112_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (512, ), (1, ))
    assert_size_stride(arg122_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (256, ), (1, ))
    assert_size_stride(arg127_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (256, ), (1, ))
    assert_size_stride(arg132_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (1024, ), (1, ))
    assert_size_stride(arg142_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, ), (1, ))
    assert_size_stride(arg151_1, (256, ), (1, ))
    assert_size_stride(arg152_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg153_1, (1024, ), (1, ))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (256, ), (1, ))
    assert_size_stride(arg161_1, (256, ), (1, ))
    assert_size_stride(arg162_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg163_1, (256, ), (1, ))
    assert_size_stride(arg164_1, (256, ), (1, ))
    assert_size_stride(arg165_1, (256, ), (1, ))
    assert_size_stride(arg166_1, (256, ), (1, ))
    assert_size_stride(arg167_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (1024, ), (1, ))
    assert_size_stride(arg172_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (256, ), (1, ))
    assert_size_stride(arg177_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg178_1, (256, ), (1, ))
    assert_size_stride(arg179_1, (256, ), (1, ))
    assert_size_stride(arg180_1, (256, ), (1, ))
    assert_size_stride(arg181_1, (256, ), (1, ))
    assert_size_stride(arg182_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (1024, ), (1, ))
    assert_size_stride(arg187_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (256, ), (1, ))
    assert_size_stride(arg196_1, (256, ), (1, ))
    assert_size_stride(arg197_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (1024, ), (1, ))
    assert_size_stride(arg202_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg203_1, (256, ), (1, ))
    assert_size_stride(arg204_1, (256, ), (1, ))
    assert_size_stride(arg205_1, (256, ), (1, ))
    assert_size_stride(arg206_1, (256, ), (1, ))
    assert_size_stride(arg207_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg208_1, (256, ), (1, ))
    assert_size_stride(arg209_1, (256, ), (1, ))
    assert_size_stride(arg210_1, (256, ), (1, ))
    assert_size_stride(arg211_1, (256, ), (1, ))
    assert_size_stride(arg212_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (1024, ), (1, ))
    assert_size_stride(arg217_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (512, ), (1, ))
    assert_size_stride(arg220_1, (512, ), (1, ))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, ), (1, ))
    assert_size_stride(arg226_1, (512, ), (1, ))
    assert_size_stride(arg227_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg228_1, (2048, ), (1, ))
    assert_size_stride(arg229_1, (2048, ), (1, ))
    assert_size_stride(arg230_1, (2048, ), (1, ))
    assert_size_stride(arg231_1, (2048, ), (1, ))
    assert_size_stride(arg232_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg233_1, (2048, ), (1, ))
    assert_size_stride(arg234_1, (2048, ), (1, ))
    assert_size_stride(arg235_1, (2048, ), (1, ))
    assert_size_stride(arg236_1, (2048, ), (1, ))
    assert_size_stride(arg237_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (512, ), (1, ))
    assert_size_stride(arg242_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg243_1, (512, ), (1, ))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (512, ), (1, ))
    assert_size_stride(arg246_1, (512, ), (1, ))
    assert_size_stride(arg247_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg248_1, (2048, ), (1, ))
    assert_size_stride(arg249_1, (2048, ), (1, ))
    assert_size_stride(arg250_1, (2048, ), (1, ))
    assert_size_stride(arg251_1, (2048, ), (1, ))
    assert_size_stride(arg252_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg253_1, (512, ), (1, ))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg258_1, (512, ), (1, ))
    assert_size_stride(arg259_1, (512, ), (1, ))
    assert_size_stride(arg260_1, (512, ), (1, ))
    assert_size_stride(arg261_1, (512, ), (1, ))
    assert_size_stride(arg262_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg263_1, (2048, ), (1, ))
    assert_size_stride(arg264_1, (2048, ), (1, ))
    assert_size_stride(arg265_1, (2048, ), (1, ))
    assert_size_stride(arg266_1, (2048, ), (1, ))
    assert_size_stride(arg267_1, (100, 2048), (2048, 1))
    assert_size_stride(arg268_1, (100, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s77, 3, 224, 224), (150528, 50176, 224, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy, aten.convolution]
        triton_poi_fused__to_copy_convolution_0_xnumel = 150528*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_0.run(arg2_1, buf0, triton_poi_fused__to_copy_convolution_0_xnumel, stream=stream0)
        del arg2_1
        buf1 = empty_strided_cuda((64, 3, 7, 7), (147, 49, 7, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_1.run(arg0_1, buf1, 9408, stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (s77, 64, 112, 112), (802816, 12544, 112, 1), 'torch.ops.aten.convolution.default')
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [bn1, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2_xnumel = 802816*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg3_1, arg4_1, arg5_1, arg6_1, triton_poi_fused__native_batch_norm_legit_no_training_relu_2_xnumel, stream=stream0)
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        buf4 = empty_strided_cuda((s77, 64, 56, 56), (200704, 3136, 56, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [bn1, relu, maxpool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3_xnumel, stream=stream0)
        del buf3
        buf5 = empty_strided_cuda((64, 64, 1, 1), (64, 1, 1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_4.run(arg7_1, buf5, 4096, stream=stream0)
        del arg7_1
        # Topologically Sorted Source Nodes: [layer1_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (s77, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_relu, layer1_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5.run(buf7, arg8_1, arg9_1, arg10_1, arg11_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel, stream=stream0)
        del arg10_1
        del arg11_1
        del arg8_1
        del arg9_1
        buf8 = empty_strided_cuda((64, 64, 3, 3), (576, 9, 3, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_relu, layer1_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_6.run(arg12_1, buf8, 36864, stream=stream0)
        del arg12_1
        # Topologically Sorted Source Nodes: [layer1_0_bn1, layer1_0_relu, layer1_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (s77, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        del buf7
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_bn2, layer1_0_relu_1, layer1_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5.run(buf10, arg13_1, arg14_1, arg15_1, arg16_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel, stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        del arg16_1
        buf11 = empty_strided_cuda((256, 64, 1, 1), (64, 1, 1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer1_0_bn2, layer1_0_relu_1, layer1_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7.run(arg17_1, buf11, 16384, stream=stream0)
        del arg17_1
        # Topologically Sorted Source Nodes: [layer1_0_bn2, layer1_0_relu_1, layer1_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (s77, 256, 56, 56), (802816, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        del buf10
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7.run(arg22_1, buf13, 16384, stream=stream0)
        del arg22_1
        # Topologically Sorted Source Nodes: [layer1_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
        buf14 = extern_kernels.convolution(buf4, buf13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (s77, 256, 56, 56), (802816, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        del buf4
        buf15 = buf12; del buf12  # reuse
        buf16 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [layer1_0_bn3, layer1_0_downsample_1, add, layer1_0_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8_xnumel = 802816*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf16, arg18_1, arg19_1, arg20_1, arg21_1, buf14, arg23_1, arg24_1, arg25_1, arg26_1, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8_xnumel, stream=stream0)
        del arg18_1
        del arg19_1
        del arg20_1
        del arg21_1
        del arg23_1
        del arg24_1
        del arg25_1
        del arg26_1
        del buf14
        buf17 = reinterpret_tensor(buf13, (64, 256, 1, 1), (256, 1, 1, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7.run(arg27_1, buf17, 16384, stream=stream0)
        del arg27_1
        # Topologically Sorted Source Nodes: [layer1_1_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf18 = extern_kernels.convolution(buf16, buf17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (s77, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_bn1, layer1_1_relu, layer1_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5.run(buf19, arg28_1, arg29_1, arg30_1, arg31_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel, stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        del arg31_1
        buf20 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_bn1, layer1_1_relu, layer1_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_6.run(arg32_1, buf20, 36864, stream=stream0)
        del arg32_1
        # Topologically Sorted Source Nodes: [layer1_1_bn1, layer1_1_relu, layer1_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (s77, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        del buf19
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_bn2, layer1_1_relu_1, layer1_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5.run(buf22, arg33_1, arg34_1, arg35_1, arg36_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel, stream=stream0)
        del arg33_1
        del arg34_1
        del arg35_1
        del arg36_1
        buf23 = reinterpret_tensor(buf17, (256, 64, 1, 1), (64, 1, 1, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_bn2, layer1_1_relu_1, layer1_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7.run(arg37_1, buf23, 16384, stream=stream0)
        del arg37_1
        # Topologically Sorted Source Nodes: [layer1_1_bn2, layer1_1_relu_1, layer1_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (s77, 256, 56, 56), (802816, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        del buf22
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [layer1_1_bn3, add_1, layer1_1_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9_xnumel = 802816*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf25, arg38_1, arg39_1, arg40_1, arg41_1, buf16, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9_xnumel, stream=stream0)
        del arg38_1
        del arg39_1
        del arg40_1
        del arg41_1
        del buf16
        buf26 = reinterpret_tensor(buf23, (64, 256, 1, 1), (256, 1, 1, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7.run(arg42_1, buf26, 16384, stream=stream0)
        del arg42_1
        # Topologically Sorted Source Nodes: [layer1_2_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf27 = extern_kernels.convolution(buf25, buf26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (s77, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        buf28 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_bn1, layer1_2_relu, layer1_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5.run(buf28, arg43_1, arg44_1, arg45_1, arg46_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel, stream=stream0)
        del arg43_1
        del arg44_1
        del arg45_1
        del arg46_1
        buf29 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_bn1, layer1_2_relu, layer1_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_6.run(arg47_1, buf29, 36864, stream=stream0)
        del arg47_1
        # Topologically Sorted Source Nodes: [layer1_2_bn1, layer1_2_relu, layer1_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (s77, 64, 56, 56), (200704, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        del buf28
        del buf29
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_bn2, layer1_2_relu_1, layer1_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5.run(buf31, arg48_1, arg49_1, arg50_1, arg51_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_5_xnumel, stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        del arg51_1
        buf32 = reinterpret_tensor(buf26, (256, 64, 1, 1), (64, 1, 1, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_bn2, layer1_2_relu_1, layer1_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_7.run(arg52_1, buf32, 16384, stream=stream0)
        del arg52_1
        # Topologically Sorted Source Nodes: [layer1_2_bn2, layer1_2_relu_1, layer1_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf33 = extern_kernels.convolution(buf31, buf32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (s77, 256, 56, 56), (802816, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        del buf31
        del buf32
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [layer1_2_bn3, add_2, layer1_2_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9_xnumel = 802816*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf34, arg53_1, arg54_1, arg55_1, arg56_1, buf25, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9_xnumel, stream=stream0)
        del arg53_1
        del arg54_1
        del arg55_1
        del arg56_1
        del buf25
        buf35 = empty_strided_cuda((128, 256, 1, 1), (256, 1, 1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_10.run(arg57_1, buf35, 32768, stream=stream0)
        del arg57_1
        # Topologically Sorted Source Nodes: [layer2_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf36 = extern_kernels.convolution(buf34, buf35, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (s77, 128, 56, 56), (401408, 3136, 56, 1), 'torch.ops.aten.convolution.default')
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_bn1, layer2_0_relu, layer2_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_11_xnumel = 401408*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_11.run(buf37, arg58_1, arg59_1, arg60_1, arg61_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_11_xnumel, stream=stream0)
        del arg58_1
        del arg59_1
        del arg60_1
        del arg61_1
        buf38 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_bn1, layer2_0_relu, layer2_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_12.run(arg62_1, buf38, 147456, stream=stream0)
        del arg62_1
        # Topologically Sorted Source Nodes: [layer2_0_bn1, layer2_0_relu, layer2_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf39 = extern_kernels.convolution(buf37, buf38, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (s77, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf37
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_relu_1, layer2_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13.run(buf40, arg63_1, arg64_1, arg65_1, arg66_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel, stream=stream0)
        del arg63_1
        del arg64_1
        del arg65_1
        del arg66_1
        buf41 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_relu_1, layer2_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14.run(arg67_1, buf41, 65536, stream=stream0)
        del arg67_1
        # Topologically Sorted Source Nodes: [layer2_0_bn2, layer2_0_relu_1, layer2_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf42 = extern_kernels.convolution(buf40, buf41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (s77, 512, 28, 28), (401408, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf40
        buf43 = empty_strided_cuda((512, 256, 1, 1), (256, 1, 1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_15.run(arg72_1, buf43, 131072, stream=stream0)
        del arg72_1
        # Topologically Sorted Source Nodes: [layer2_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
        buf44 = extern_kernels.convolution(buf34, buf43, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (s77, 512, 28, 28), (401408, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf34
        buf45 = buf42; del buf42  # reuse
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [layer2_0_bn3, layer2_0_downsample_1, add_3, layer2_0_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16_xnumel = 401408*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf46, arg68_1, arg69_1, arg70_1, arg71_1, buf44, arg73_1, arg74_1, arg75_1, arg76_1, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16_xnumel, stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        del arg71_1
        del arg73_1
        del arg74_1
        del arg75_1
        del arg76_1
        del buf44
        buf47 = reinterpret_tensor(buf41, (128, 512, 1, 1), (512, 1, 1, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14.run(arg77_1, buf47, 65536, stream=stream0)
        del arg77_1
        # Topologically Sorted Source Nodes: [layer2_1_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf48 = extern_kernels.convolution(buf46, buf47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (s77, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_bn1, layer2_1_relu, layer2_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13.run(buf49, arg78_1, arg79_1, arg80_1, arg81_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel, stream=stream0)
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        buf50 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_bn1, layer2_1_relu, layer2_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_12.run(arg82_1, buf50, 147456, stream=stream0)
        del arg82_1
        # Topologically Sorted Source Nodes: [layer2_1_bn1, layer2_1_relu, layer2_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf51 = extern_kernels.convolution(buf49, buf50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (s77, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf49
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_bn2, layer2_1_relu_1, layer2_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13.run(buf52, arg83_1, arg84_1, arg85_1, arg86_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel, stream=stream0)
        del arg83_1
        del arg84_1
        del arg85_1
        del arg86_1
        buf53 = reinterpret_tensor(buf47, (512, 128, 1, 1), (128, 1, 1, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_bn2, layer2_1_relu_1, layer2_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14.run(arg87_1, buf53, 65536, stream=stream0)
        del arg87_1
        # Topologically Sorted Source Nodes: [layer2_1_bn2, layer2_1_relu_1, layer2_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (s77, 512, 28, 28), (401408, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf52
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [layer2_1_bn3, add_4, layer2_1_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17_xnumel = 401408*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf55, arg88_1, arg89_1, arg90_1, arg91_1, buf46, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17_xnumel, stream=stream0)
        del arg88_1
        del arg89_1
        del arg90_1
        del arg91_1
        del buf46
        buf56 = reinterpret_tensor(buf53, (128, 512, 1, 1), (512, 1, 1, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14.run(arg92_1, buf56, 65536, stream=stream0)
        del arg92_1
        # Topologically Sorted Source Nodes: [layer2_2_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf57 = extern_kernels.convolution(buf55, buf56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (s77, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
        buf58 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_bn1, layer2_2_relu, layer2_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13.run(buf58, arg93_1, arg94_1, arg95_1, arg96_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel, stream=stream0)
        del arg93_1
        del arg94_1
        del arg95_1
        del arg96_1
        buf59 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_bn1, layer2_2_relu, layer2_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_12.run(arg97_1, buf59, 147456, stream=stream0)
        del arg97_1
        # Topologically Sorted Source Nodes: [layer2_2_bn1, layer2_2_relu, layer2_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf60 = extern_kernels.convolution(buf58, buf59, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (s77, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf58
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_bn2, layer2_2_relu_1, layer2_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13.run(buf61, arg98_1, arg99_1, arg100_1, arg101_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel, stream=stream0)
        del arg100_1
        del arg101_1
        del arg98_1
        del arg99_1
        buf62 = reinterpret_tensor(buf56, (512, 128, 1, 1), (128, 1, 1, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_bn2, layer2_2_relu_1, layer2_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14.run(arg102_1, buf62, 65536, stream=stream0)
        del arg102_1
        # Topologically Sorted Source Nodes: [layer2_2_bn2, layer2_2_relu_1, layer2_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf63 = extern_kernels.convolution(buf61, buf62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (s77, 512, 28, 28), (401408, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf61
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [layer2_2_bn3, add_5, layer2_2_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17_xnumel = 401408*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf64, arg103_1, arg104_1, arg105_1, arg106_1, buf55, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17_xnumel, stream=stream0)
        del arg103_1
        del arg104_1
        del arg105_1
        del arg106_1
        del buf55
        buf65 = reinterpret_tensor(buf62, (128, 512, 1, 1), (512, 1, 1, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14.run(arg107_1, buf65, 65536, stream=stream0)
        del arg107_1
        # Topologically Sorted Source Nodes: [layer2_3_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (s77, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_bn1, layer2_3_relu, layer2_3_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13.run(buf67, arg108_1, arg109_1, arg110_1, arg111_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel, stream=stream0)
        del arg108_1
        del arg109_1
        del arg110_1
        del arg111_1
        buf68 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_bn1, layer2_3_relu, layer2_3_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_12.run(arg112_1, buf68, 147456, stream=stream0)
        del arg112_1
        # Topologically Sorted Source Nodes: [layer2_3_bn1, layer2_3_relu, layer2_3_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf69 = extern_kernels.convolution(buf67, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (s77, 128, 28, 28), (100352, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf67
        del buf68
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_bn2, layer2_3_relu_1, layer2_3_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13.run(buf70, arg113_1, arg114_1, arg115_1, arg116_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_13_xnumel, stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        del arg116_1
        buf71 = reinterpret_tensor(buf65, (512, 128, 1, 1), (128, 1, 1, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_bn2, layer2_3_relu_1, layer2_3_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_14.run(arg117_1, buf71, 65536, stream=stream0)
        del arg117_1
        # Topologically Sorted Source Nodes: [layer2_3_bn2, layer2_3_relu_1, layer2_3_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf72 = extern_kernels.convolution(buf70, buf71, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (s77, 512, 28, 28), (401408, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf70
        del buf71
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [layer2_3_bn3, add_6, layer2_3_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17_xnumel = 401408*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf73, arg118_1, arg119_1, arg120_1, arg121_1, buf64, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17_xnumel, stream=stream0)
        del arg118_1
        del arg119_1
        del arg120_1
        del arg121_1
        del buf64
        buf74 = reinterpret_tensor(buf43, (256, 512, 1, 1), (512, 1, 1, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_15.run(arg122_1, buf74, 131072, stream=stream0)
        del arg122_1
        # Topologically Sorted Source Nodes: [layer3_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf75 = extern_kernels.convolution(buf73, buf74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (s77, 256, 28, 28), (200704, 784, 28, 1), 'torch.ops.aten.convolution.default')
        del buf74
        buf76 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_bn1, layer3_0_relu, layer3_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_18_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_18.run(buf76, arg123_1, arg124_1, arg125_1, arg126_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_18_xnumel, stream=stream0)
        del arg123_1
        del arg124_1
        del arg125_1
        del arg126_1
        buf77 = empty_strided_cuda((256, 256, 3, 3), (2304, 9, 3, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_bn1, layer3_0_relu, layer3_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19.run(arg127_1, buf77, 589824, stream=stream0)
        del arg127_1
        # Topologically Sorted Source Nodes: [layer3_0_bn1, layer3_0_relu, layer3_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf78 = extern_kernels.convolution(buf76, buf77, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf76
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_relu_1, layer3_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf79, arg128_1, arg129_1, arg130_1, arg131_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg128_1
        del arg129_1
        del arg130_1
        del arg131_1
        buf80 = empty_strided_cuda((1024, 256, 1, 1), (256, 1, 1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_relu_1, layer3_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg132_1, buf80, 262144, stream=stream0)
        del arg132_1
        # Topologically Sorted Source Nodes: [layer3_0_bn2, layer3_0_relu_1, layer3_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf81 = extern_kernels.convolution(buf79, buf80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (s77, 1024, 14, 14), (200704, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf79
        buf82 = empty_strided_cuda((1024, 512, 1, 1), (512, 1, 1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_22.run(arg137_1, buf82, 524288, stream=stream0)
        del arg137_1
        # Topologically Sorted Source Nodes: [layer3_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
        buf83 = extern_kernels.convolution(buf73, buf82, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (s77, 1024, 14, 14), (200704, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf73
        buf84 = buf81; del buf81  # reuse
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [layer3_0_bn3, layer3_0_downsample_1, add_7, layer3_0_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf85, arg133_1, arg134_1, arg135_1, arg136_1, buf83, arg138_1, arg139_1, arg140_1, arg141_1, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23_xnumel, stream=stream0)
        del arg133_1
        del arg134_1
        del arg135_1
        del arg136_1
        del arg138_1
        del arg139_1
        del arg140_1
        del arg141_1
        del buf83
        buf86 = reinterpret_tensor(buf80, (256, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg142_1, buf86, 262144, stream=stream0)
        del arg142_1
        # Topologically Sorted Source Nodes: [layer3_1_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf87 = extern_kernels.convolution(buf85, buf86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        buf88 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_bn1, layer3_1_relu, layer3_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf88, arg143_1, arg144_1, arg145_1, arg146_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg143_1
        del arg144_1
        del arg145_1
        del arg146_1
        buf89 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_bn1, layer3_1_relu, layer3_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19.run(arg147_1, buf89, 589824, stream=stream0)
        del arg147_1
        # Topologically Sorted Source Nodes: [layer3_1_bn1, layer3_1_relu, layer3_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf90 = extern_kernels.convolution(buf88, buf89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf88
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_bn2, layer3_1_relu_1, layer3_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf91, arg148_1, arg149_1, arg150_1, arg151_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg148_1
        del arg149_1
        del arg150_1
        del arg151_1
        buf92 = reinterpret_tensor(buf86, (1024, 256, 1, 1), (256, 1, 1, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_bn2, layer3_1_relu_1, layer3_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg152_1, buf92, 262144, stream=stream0)
        del arg152_1
        # Topologically Sorted Source Nodes: [layer3_1_bn2, layer3_1_relu_1, layer3_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf93 = extern_kernels.convolution(buf91, buf92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (s77, 1024, 14, 14), (200704, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf91
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [layer3_1_bn3, add_8, layer3_1_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf94, arg153_1, arg154_1, arg155_1, arg156_1, buf85, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel, stream=stream0)
        del arg153_1
        del arg154_1
        del arg155_1
        del arg156_1
        del buf85
        buf95 = reinterpret_tensor(buf92, (256, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg157_1, buf95, 262144, stream=stream0)
        del arg157_1
        # Topologically Sorted Source Nodes: [layer3_2_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf96 = extern_kernels.convolution(buf94, buf95, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_bn1, layer3_2_relu, layer3_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf97, arg158_1, arg159_1, arg160_1, arg161_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg158_1
        del arg159_1
        del arg160_1
        del arg161_1
        buf98 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_bn1, layer3_2_relu, layer3_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19.run(arg162_1, buf98, 589824, stream=stream0)
        del arg162_1
        # Topologically Sorted Source Nodes: [layer3_2_bn1, layer3_2_relu, layer3_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf99 = extern_kernels.convolution(buf97, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf97
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_bn2, layer3_2_relu_1, layer3_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf100, arg163_1, arg164_1, arg165_1, arg166_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg163_1
        del arg164_1
        del arg165_1
        del arg166_1
        buf101 = reinterpret_tensor(buf95, (1024, 256, 1, 1), (256, 1, 1, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_bn2, layer3_2_relu_1, layer3_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg167_1, buf101, 262144, stream=stream0)
        del arg167_1
        # Topologically Sorted Source Nodes: [layer3_2_bn2, layer3_2_relu_1, layer3_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf102 = extern_kernels.convolution(buf100, buf101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (s77, 1024, 14, 14), (200704, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf100
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [layer3_2_bn3, add_9, layer3_2_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf103, arg168_1, arg169_1, arg170_1, arg171_1, buf94, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel, stream=stream0)
        del arg168_1
        del arg169_1
        del arg170_1
        del arg171_1
        del buf94
        buf104 = reinterpret_tensor(buf101, (256, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg172_1, buf104, 262144, stream=stream0)
        del arg172_1
        # Topologically Sorted Source Nodes: [layer3_3_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf105 = extern_kernels.convolution(buf103, buf104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_bn1, layer3_3_relu, layer3_3_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf106, arg173_1, arg174_1, arg175_1, arg176_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg173_1
        del arg174_1
        del arg175_1
        del arg176_1
        buf107 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_bn1, layer3_3_relu, layer3_3_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19.run(arg177_1, buf107, 589824, stream=stream0)
        del arg177_1
        # Topologically Sorted Source Nodes: [layer3_3_bn1, layer3_3_relu, layer3_3_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf108 = extern_kernels.convolution(buf106, buf107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf106
        buf109 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_bn2, layer3_3_relu_1, layer3_3_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf109, arg178_1, arg179_1, arg180_1, arg181_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg178_1
        del arg179_1
        del arg180_1
        del arg181_1
        buf110 = reinterpret_tensor(buf104, (1024, 256, 1, 1), (256, 1, 1, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_bn2, layer3_3_relu_1, layer3_3_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg182_1, buf110, 262144, stream=stream0)
        del arg182_1
        # Topologically Sorted Source Nodes: [layer3_3_bn2, layer3_3_relu_1, layer3_3_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf111 = extern_kernels.convolution(buf109, buf110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (s77, 1024, 14, 14), (200704, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf109
        buf112 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [layer3_3_bn3, add_10, layer3_3_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf112, arg183_1, arg184_1, arg185_1, arg186_1, buf103, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel, stream=stream0)
        del arg183_1
        del arg184_1
        del arg185_1
        del arg186_1
        del buf103
        buf113 = reinterpret_tensor(buf110, (256, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg187_1, buf113, 262144, stream=stream0)
        del arg187_1
        # Topologically Sorted Source Nodes: [layer3_4_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf114 = extern_kernels.convolution(buf112, buf113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_bn1, layer3_4_relu, layer3_4_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf115, arg188_1, arg189_1, arg190_1, arg191_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        del arg191_1
        buf116 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_bn1, layer3_4_relu, layer3_4_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19.run(arg192_1, buf116, 589824, stream=stream0)
        del arg192_1
        # Topologically Sorted Source Nodes: [layer3_4_bn1, layer3_4_relu, layer3_4_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf117 = extern_kernels.convolution(buf115, buf116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf115
        buf118 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_bn2, layer3_4_relu_1, layer3_4_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf118, arg193_1, arg194_1, arg195_1, arg196_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg193_1
        del arg194_1
        del arg195_1
        del arg196_1
        buf119 = reinterpret_tensor(buf113, (1024, 256, 1, 1), (256, 1, 1, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_bn2, layer3_4_relu_1, layer3_4_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg197_1, buf119, 262144, stream=stream0)
        del arg197_1
        # Topologically Sorted Source Nodes: [layer3_4_bn2, layer3_4_relu_1, layer3_4_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf120 = extern_kernels.convolution(buf118, buf119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (s77, 1024, 14, 14), (200704, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf118
        buf121 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [layer3_4_bn3, add_11, layer3_4_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf121, arg198_1, arg199_1, arg200_1, arg201_1, buf112, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel, stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        del arg201_1
        del buf112
        buf122 = reinterpret_tensor(buf119, (256, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg202_1, buf122, 262144, stream=stream0)
        del arg202_1
        # Topologically Sorted Source Nodes: [layer3_5_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf123 = extern_kernels.convolution(buf121, buf122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_bn1, layer3_5_relu, layer3_5_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf124, arg203_1, arg204_1, arg205_1, arg206_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg203_1
        del arg204_1
        del arg205_1
        del arg206_1
        buf125 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_bn1, layer3_5_relu, layer3_5_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_19.run(arg207_1, buf125, 589824, stream=stream0)
        del arg207_1
        # Topologically Sorted Source Nodes: [layer3_5_bn1, layer3_5_relu, layer3_5_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf126 = extern_kernels.convolution(buf124, buf125, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (s77, 256, 14, 14), (50176, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf124
        del buf125
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_bn2, layer3_5_relu_1, layer3_5_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel = 50176*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20.run(buf127, arg208_1, arg209_1, arg210_1, arg211_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_20_xnumel, stream=stream0)
        del arg208_1
        del arg209_1
        del arg210_1
        del arg211_1
        buf128 = reinterpret_tensor(buf122, (1024, 256, 1, 1), (256, 1, 1, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_bn2, layer3_5_relu_1, layer3_5_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_21.run(arg212_1, buf128, 262144, stream=stream0)
        del arg212_1
        # Topologically Sorted Source Nodes: [layer3_5_bn2, layer3_5_relu_1, layer3_5_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf129 = extern_kernels.convolution(buf127, buf128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (s77, 1024, 14, 14), (200704, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf127
        del buf128
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [layer3_5_bn3, add_12, layer3_5_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel = 200704*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf130, arg213_1, arg214_1, arg215_1, arg216_1, buf121, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24_xnumel, stream=stream0)
        del arg213_1
        del arg214_1
        del arg215_1
        del arg216_1
        del buf121
        buf131 = reinterpret_tensor(buf82, (512, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_22.run(arg217_1, buf131, 524288, stream=stream0)
        del arg217_1
        # Topologically Sorted Source Nodes: [layer4_0_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf132 = extern_kernels.convolution(buf130, buf131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (s77, 512, 14, 14), (100352, 196, 14, 1), 'torch.ops.aten.convolution.default')
        del buf131
        buf133 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_bn1, layer4_0_relu, layer4_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_25_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_25.run(buf133, arg218_1, arg219_1, arg220_1, arg221_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_25_xnumel, stream=stream0)
        del arg218_1
        del arg219_1
        del arg220_1
        del arg221_1
        buf134 = empty_strided_cuda((512, 512, 3, 3), (4608, 9, 3, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_bn1, layer4_0_relu, layer4_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_26.run(arg222_1, buf134, 2359296, stream=stream0)
        del arg222_1
        # Topologically Sorted Source Nodes: [layer4_0_bn1, layer4_0_relu, layer4_0_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf135 = extern_kernels.convolution(buf133, buf134, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (s77, 512, 7, 7), (25088, 49, 7, 1), 'torch.ops.aten.convolution.default')
        del buf133
        buf136 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_relu_1, layer4_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel = 25088*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27.run(buf136, arg223_1, arg224_1, arg225_1, arg226_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel, stream=stream0)
        del arg223_1
        del arg224_1
        del arg225_1
        del arg226_1
        buf137 = empty_strided_cuda((2048, 512, 1, 1), (512, 1, 1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_relu_1, layer4_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_28.run(arg227_1, buf137, 1048576, stream=stream0)
        del arg227_1
        # Topologically Sorted Source Nodes: [layer4_0_bn2, layer4_0_relu_1, layer4_0_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf138 = extern_kernels.convolution(buf136, buf137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (s77, 2048, 7, 7), (100352, 49, 7, 1), 'torch.ops.aten.convolution.default')
        del buf136
        buf139 = empty_strided_cuda((2048, 1024, 1, 1), (1024, 1, 1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_convolution_29.run(arg232_1, buf139, 2097152, stream=stream0)
        del arg232_1
        # Topologically Sorted Source Nodes: [layer4_0_downsample_0], Original ATen: [aten._to_copy, aten.convolution]
        buf140 = extern_kernels.convolution(buf130, buf139, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (s77, 2048, 7, 7), (100352, 49, 7, 1), 'torch.ops.aten.convolution.default')
        del buf130
        del buf139
        buf141 = buf138; del buf138  # reuse
        buf142 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [layer4_0_bn3, layer4_0_downsample_1, add_13, layer4_0_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30.run(buf142, arg228_1, arg229_1, arg230_1, arg231_1, buf140, arg233_1, arg234_1, arg235_1, arg236_1, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30_xnumel, stream=stream0)
        del arg228_1
        del arg229_1
        del arg230_1
        del arg231_1
        del arg233_1
        del arg234_1
        del arg235_1
        del arg236_1
        del buf140
        buf143 = reinterpret_tensor(buf137, (512, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_28.run(arg237_1, buf143, 1048576, stream=stream0)
        del arg237_1
        # Topologically Sorted Source Nodes: [layer4_1_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf144 = extern_kernels.convolution(buf142, buf143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (s77, 512, 7, 7), (25088, 49, 7, 1), 'torch.ops.aten.convolution.default')
        buf145 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_bn1, layer4_1_relu, layer4_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel = 25088*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27.run(buf145, arg238_1, arg239_1, arg240_1, arg241_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel, stream=stream0)
        del arg238_1
        del arg239_1
        del arg240_1
        del arg241_1
        buf146 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_bn1, layer4_1_relu, layer4_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_26.run(arg242_1, buf146, 2359296, stream=stream0)
        del arg242_1
        # Topologically Sorted Source Nodes: [layer4_1_bn1, layer4_1_relu, layer4_1_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf147 = extern_kernels.convolution(buf145, buf146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (s77, 512, 7, 7), (25088, 49, 7, 1), 'torch.ops.aten.convolution.default')
        del buf145
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_bn2, layer4_1_relu_1, layer4_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel = 25088*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27.run(buf148, arg243_1, arg244_1, arg245_1, arg246_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel, stream=stream0)
        del arg243_1
        del arg244_1
        del arg245_1
        del arg246_1
        buf149 = reinterpret_tensor(buf143, (2048, 512, 1, 1), (512, 1, 1, 1), 0); del buf143  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_bn2, layer4_1_relu_1, layer4_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_28.run(arg247_1, buf149, 1048576, stream=stream0)
        del arg247_1
        # Topologically Sorted Source Nodes: [layer4_1_bn2, layer4_1_relu_1, layer4_1_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf150 = extern_kernels.convolution(buf148, buf149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (s77, 2048, 7, 7), (100352, 49, 7, 1), 'torch.ops.aten.convolution.default')
        del buf148
        buf151 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [layer4_1_bn3, add_14, layer4_1_relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31_xnumel = 100352*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31.run(buf151, arg248_1, arg249_1, arg250_1, arg251_1, buf142, triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31_xnumel, stream=stream0)
        del arg248_1
        del arg249_1
        del arg250_1
        del arg251_1
        del buf142
        buf152 = reinterpret_tensor(buf149, (512, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_28.run(arg252_1, buf152, 1048576, stream=stream0)
        del arg252_1
        # Topologically Sorted Source Nodes: [layer4_2_conv1], Original ATen: [aten._to_copy, aten.convolution]
        buf153 = extern_kernels.convolution(buf151, buf152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (s77, 512, 7, 7), (25088, 49, 7, 1), 'torch.ops.aten.convolution.default')
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_bn1, layer4_2_relu, layer4_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel = 25088*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27.run(buf154, arg253_1, arg254_1, arg255_1, arg256_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel, stream=stream0)
        del arg253_1
        del arg254_1
        del arg255_1
        del arg256_1
        buf155 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_bn1, layer4_2_relu, layer4_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_26.run(arg257_1, buf155, 2359296, stream=stream0)
        del arg257_1
        # Topologically Sorted Source Nodes: [layer4_2_bn1, layer4_2_relu, layer4_2_conv2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf156 = extern_kernels.convolution(buf154, buf155, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (s77, 512, 7, 7), (25088, 49, 7, 1), 'torch.ops.aten.convolution.default')
        del buf154
        del buf155
        buf157 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_bn2, layer4_2_relu_1, layer4_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel = 25088*s77
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27.run(buf157, arg258_1, arg259_1, arg260_1, arg261_1, triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_27_xnumel, stream=stream0)
        del arg258_1
        del arg259_1
        del arg260_1
        del arg261_1
        buf158 = reinterpret_tensor(buf152, (2048, 512, 1, 1), (512, 1, 1, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [layer4_2_bn2, layer4_2_relu_1, layer4_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__to_copy_convolution_relu_28.run(arg262_1, buf158, 1048576, stream=stream0)
        del arg262_1
        # Topologically Sorted Source Nodes: [layer4_2_bn2, layer4_2_relu_1, layer4_2_conv3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._to_copy, aten.convolution]
        buf159 = extern_kernels.convolution(buf157, buf158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (s77, 2048, 7, 7), (100352, 49, 7, 1), 'torch.ops.aten.convolution.default')
        del buf157
        del buf158
        buf161 = empty_strided_cuda((s77, 2048, 1, 1), (2048, 1, 2048*s77, 2048*s77), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer4_2_bn3, add_15, layer4_2_relu_2, avgpool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_32_xnumel = 2048*s77
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_32.run(buf159, arg263_1, arg264_1, arg265_1, arg266_1, buf151, buf161, triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_32_xnumel, 49, stream=stream0)
        del arg263_1
        del arg264_1
        del arg265_1
        del arg266_1
        del buf151
        del buf159
        buf162 = empty_strided_cuda((100, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [fc], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_33.run(arg267_1, buf162, 204800, stream=stream0)
        del arg267_1
        buf163 = empty_strided_cuda((100, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [fc], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_34.run(arg268_1, buf163, 100, stream=stream0)
        del arg268_1
        buf164 = empty_strided_cuda((s77, 100), (100, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [fc], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf163, reinterpret_tensor(buf161, (s77, 2048), (2048, 1), 0), reinterpret_tensor(buf162, (2048, 100), (1, 2048), 0), alpha=1, beta=1, out=buf164)
        del buf161
        del buf162
        del buf163
    return (buf164, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = 64
    arg2_1 = rand_strided((64, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((100, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
