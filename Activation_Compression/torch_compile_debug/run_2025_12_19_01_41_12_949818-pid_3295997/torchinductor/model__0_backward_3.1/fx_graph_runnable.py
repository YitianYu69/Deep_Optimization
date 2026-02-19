
import os
os.environ['TORCH_LOGS'] = 'inductor'
os.environ['TORCHINDUCTOR_TRACE'] = '1'
os.environ['TORCHINDUCTOR_VERBOSE'] = '1'
os.environ['TORCHINDUCTOR_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_DUMP'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_yyu496'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_yyu496/triton/0'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.unroll_reductions_threshold = 8
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False



isolate_fails_code_str = None




# torch version: 2.8.0+cu128
# torch cuda version: 12.8
# torch git version: a1cb3cc05d46d198467bebbb6e8fba50a325d4e7


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Tue_May_27_02:21:03_PDT_2025 
# Cuda compilation tools, release 12.9, V12.9.86 
# Build cuda_12.9.r12.9/compiler.36037853_0 

# GPU Hardware Info: 
# NVIDIA H100 80GB HBM3 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_4, primals_10, primals_16, primals_22, primals_28, primals_34, primals_40, primals_46, primals_52, primals_58, primals_64, primals_70, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_268, primals_274, primals_280, primals_286, primals_292, primals_298, primals_304, primals_310, primals_316, getitem, getitem_1, getitem_2, rsqrt, getitem_7, getitem_8, getitem_9, getitem_10, getitem_12, getitem_14, convert_element_type_2, getitem_15, getitem_16, getitem_17, rsqrt_1, getitem_22, getitem_23, getitem_24, getitem_27, convert_element_type_3, getitem_28, getitem_29, getitem_30, rsqrt_2, getitem_35, getitem_36, getitem_37, getitem_40, convert_element_type_4, getitem_41, getitem_42, getitem_43, rsqrt_3, getitem_48, getitem_49, getitem_50, convert_element_type_5, getitem_51, getitem_52, getitem_53, rsqrt_4, getitem_58, getitem_59, getitem_60, getitem_63, convert_element_type_6, getitem_64, getitem_65, getitem_66, rsqrt_5, getitem_71, getitem_72, getitem_73, getitem_76, convert_element_type_7, getitem_77, getitem_78, getitem_79, rsqrt_6, getitem_84, getitem_85, getitem_86, getitem_89, convert_element_type_8, getitem_90, getitem_91, getitem_92, rsqrt_7, getitem_97, getitem_98, getitem_99, getitem_102, convert_element_type_9, getitem_103, getitem_104, getitem_105, rsqrt_8, getitem_110, getitem_111, getitem_112, getitem_115, convert_element_type_10, getitem_116, getitem_117, getitem_118, rsqrt_9, getitem_123, getitem_124, getitem_125, getitem_128, convert_element_type_11, getitem_129, getitem_130, getitem_131, rsqrt_10, getitem_136, getitem_137, getitem_138, getitem_141, convert_element_type_12, getitem_142, getitem_143, getitem_144, rsqrt_11, getitem_149, getitem_150, getitem_151, getitem_154, convert_element_type_13, getitem_155, getitem_156, getitem_157, rsqrt_12, getitem_162, getitem_163, getitem_164, getitem_167, convert_element_type_14, getitem_168, getitem_169, getitem_170, rsqrt_13, getitem_175, getitem_176, getitem_177, convert_element_type_15, getitem_178, getitem_179, getitem_180, rsqrt_14, getitem_185, getitem_186, getitem_187, getitem_190, convert_element_type_16, getitem_191, getitem_192, getitem_193, rsqrt_15, getitem_198, getitem_199, getitem_200, getitem_203, convert_element_type_17, getitem_204, getitem_205, getitem_206, rsqrt_16, getitem_211, getitem_212, getitem_213, getitem_216, convert_element_type_18, getitem_217, getitem_218, getitem_219, rsqrt_17, getitem_224, getitem_225, getitem_226, getitem_229, convert_element_type_19, getitem_230, getitem_231, getitem_232, rsqrt_18, getitem_237, getitem_238, getitem_239, getitem_242, convert_element_type_20, getitem_243, getitem_244, getitem_245, rsqrt_19, getitem_250, getitem_251, getitem_252, getitem_255, convert_element_type_21, getitem_256, getitem_257, getitem_258, rsqrt_20, getitem_263, getitem_264, getitem_265, getitem_268, convert_element_type_22, getitem_269, getitem_270, getitem_271, rsqrt_21, getitem_276, getitem_277, getitem_278, getitem_281, convert_element_type_23, getitem_282, getitem_283, getitem_284, rsqrt_22, getitem_289, getitem_290, getitem_291, getitem_294, convert_element_type_24, getitem_295, getitem_296, getitem_297, rsqrt_23, getitem_302, getitem_303, getitem_304, getitem_307, convert_element_type_25, getitem_308, getitem_309, getitem_310, rsqrt_24, getitem_315, getitem_316, getitem_317, getitem_320, convert_element_type_26, getitem_321, getitem_322, getitem_323, rsqrt_25, getitem_328, getitem_329, getitem_330, getitem_333, convert_element_type_27, getitem_334, getitem_335, getitem_336, rsqrt_26, getitem_341, getitem_342, getitem_343, convert_element_type_28, getitem_344, getitem_345, getitem_346, rsqrt_27, getitem_351, getitem_352, getitem_353, getitem_356, convert_element_type_29, getitem_357, getitem_358, getitem_359, rsqrt_28, getitem_364, getitem_365, getitem_366, getitem_369, convert_element_type_30, getitem_370, getitem_371, getitem_372, rsqrt_29, getitem_377, getitem_378, getitem_379, getitem_382, convert_element_type_31, getitem_383, getitem_384, getitem_385, rsqrt_30, getitem_390, getitem_391, getitem_392, getitem_395, convert_element_type_32, getitem_396, getitem_397, getitem_398, rsqrt_31, getitem_403, getitem_404, getitem_405, getitem_408, convert_element_type_33, getitem_409, getitem_410, getitem_411, rsqrt_32, getitem_416, getitem_417, getitem_418, getitem_421, convert_element_type_34, getitem_422, getitem_423, getitem_424, rsqrt_33, getitem_429, getitem_430, getitem_431, getitem_434, convert_element_type_35, getitem_435, getitem_436, getitem_437, rsqrt_34, getitem_442, getitem_443, getitem_444, getitem_447, convert_element_type_36, getitem_448, getitem_449, getitem_450, rsqrt_35, getitem_455, getitem_456, getitem_457, getitem_460, convert_element_type_37, getitem_461, getitem_462, getitem_463, rsqrt_36, getitem_468, getitem_469, getitem_470, getitem_473, convert_element_type_38, getitem_474, getitem_475, getitem_476, rsqrt_37, getitem_481, getitem_482, getitem_483, getitem_486, convert_element_type_39, getitem_487, getitem_488, getitem_489, rsqrt_38, getitem_494, getitem_495, getitem_496, getitem_499, convert_element_type_40, getitem_500, getitem_501, getitem_502, rsqrt_39, getitem_507, getitem_508, getitem_509, getitem_512, convert_element_type_41, getitem_513, getitem_514, getitem_515, rsqrt_40, getitem_520, getitem_521, getitem_522, getitem_525, convert_element_type_42, getitem_526, getitem_527, getitem_528, rsqrt_41, getitem_533, getitem_534, getitem_535, getitem_538, convert_element_type_43, getitem_539, getitem_540, getitem_541, rsqrt_42, getitem_546, getitem_547, getitem_548, getitem_551, convert_element_type_44, getitem_552, getitem_553, getitem_554, rsqrt_43, getitem_559, getitem_560, getitem_561, getitem_564, convert_element_type_45, getitem_565, getitem_566, getitem_567, rsqrt_44, getitem_572, getitem_573, getitem_574, getitem_577, convert_element_type_46, getitem_578, getitem_579, getitem_580, rsqrt_45, getitem_585, getitem_586, getitem_587, convert_element_type_47, getitem_588, getitem_589, getitem_590, rsqrt_46, getitem_595, getitem_596, getitem_597, getitem_600, convert_element_type_48, getitem_601, getitem_602, getitem_603, rsqrt_47, getitem_608, getitem_609, getitem_610, getitem_613, convert_element_type_49, getitem_614, getitem_615, getitem_616, rsqrt_48, getitem_621, getitem_622, getitem_623, getitem_626, convert_element_type_50, getitem_627, getitem_628, getitem_629, rsqrt_49, getitem_634, getitem_635, getitem_636, getitem_639, convert_element_type_51, getitem_640, getitem_641, getitem_642, rsqrt_50, getitem_647, getitem_648, getitem_649, getitem_652, convert_element_type_52, getitem_653, getitem_654, getitem_655, rsqrt_51, getitem_660, getitem_661, getitem_662, getitem_665, convert_element_type_53, getitem_666, getitem_667, getitem_668, rsqrt_52, getitem_673, getitem_674, getitem_675, getitem_678, getitem_679, getitem_680, getitem_681, convert_element_type_54, tangents_1):
        permute_156 = torch.ops.aten.permute.default(tangents_1, [1, 0])
        empty_476 = torch.ops.aten.empty.memory_format([2048, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_311 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 432, constant_args_idx = 622, grid = [(2048, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_679, 'S_ptr': getitem_680, 'M_ptr': getitem_681, 'Y_ptr': empty_476, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_679 = getitem_680 = getitem_681 = empty_476 = None
        getitem_682 = triton_kernel_wrapper_functional_proxy_311['Y_ptr'];  triton_kernel_wrapper_functional_proxy_311 = None
        view_1326 = torch.ops.aten.view.default(getitem_682, [512, 4, 512]);  getitem_682 = None
        view_1327 = torch.ops.aten.view.default(view_1326, [512, 2048]);  view_1326 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_156, [0, 0, 0, 4]);  permute_156 = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default, view_1327);  constant_pad_nd_default = view_1327 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -4);  mm_default = None
        mm_2 = torch.ops.aten.mm.default(tangents_1, convert_element_type_54);  tangents_1 = convert_element_type_54 = None
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        view_1330 = torch.ops.aten.view.default(mm_2, [512, 2048, 1, 1]);  mm_2 = None
        expand = torch.ops.aten.expand.default(view_1330, [512, 2048, 7, 7]);  view_1330 = None
        div_159 = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        full_default_310 = torch.ops.aten.full.default([100352, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_312 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 623, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_678, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_678 = None
        getitem_683 = triton_kernel_wrapper_functional_proxy_312['Y_ptr'];  triton_kernel_wrapper_functional_proxy_312 = None
        view_1333 = torch.ops.aten.view.default(getitem_683, [512, 196, 512]);  getitem_683 = None
        view_1334 = torch.ops.aten.view.default(view_1333, [512, 2048, 7, 7]);  view_1333 = None
        mul_371 = torch.ops.aten.mul.Tensor(div_159, view_1334);  div_159 = view_1334 = None
        empty_477 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_313 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 434, constant_args_idx = 624, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_673, 'S_ptr': getitem_674, 'M_ptr': getitem_675, 'Y_ptr': empty_477, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_673 = getitem_674 = getitem_675 = empty_477 = None
        getitem_684 = triton_kernel_wrapper_functional_proxy_313['Y_ptr'];  triton_kernel_wrapper_functional_proxy_313 = None
        view_1349 = torch.ops.aten.view.default(mul_371, [512, 2048, 49])
        view_1350 = torch.ops.aten.view.default(getitem_684, [512, 196, 512]);  getitem_684 = None
        view_1351 = torch.ops.aten.view.default(view_1350, [512, 2048, 49]);  view_1350 = None
        full_default_264 = torch.ops.aten.full.default([2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_314 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 436, constant_args_idx = 625, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1351, 'DY': view_1349, 'DBETA': full_default_264, 'DGAMMA': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_685 = triton_kernel_wrapper_functional_proxy_314['DBETA']
        getitem_686 = triton_kernel_wrapper_functional_proxy_314['DGAMMA'];  triton_kernel_wrapper_functional_proxy_314 = None
        empty_478 = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_157 = torch.ops.aten.permute.default(empty_478, [0, 1, 2]);  empty_478 = None
        triton_kernel_wrapper_functional_proxy_315 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 438, constant_args_idx = 626, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1351, 'DY': view_1349, 'INVSTD': rsqrt_52, 'GAMMA': primals_316, 'DBETA': getitem_685, 'DGAMMA': getitem_686, 'DX': permute_157, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1351 = view_1349 = rsqrt_52 = primals_316 = permute_157 = None
        getitem_687 = triton_kernel_wrapper_functional_proxy_315['DX'];  triton_kernel_wrapper_functional_proxy_315 = None
        convert_element_type_default_105 = torch.ops.prims.convert_element_type.default(getitem_686, torch.float32);  getitem_686 = None
        convert_element_type_default_104 = torch.ops.prims.convert_element_type.default(getitem_685, torch.float32);  getitem_685 = None
        empty_479 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_316 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 439, constant_args_idx = 627, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_666, 'S_ptr': getitem_667, 'M_ptr': getitem_668, 'Y_ptr': empty_479, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_666 = getitem_667 = getitem_668 = empty_479 = None
        getitem_688 = triton_kernel_wrapper_functional_proxy_316['Y_ptr'];  triton_kernel_wrapper_functional_proxy_316 = None
        view_1367 = torch.ops.aten.view.default(getitem_687, [512, 2048, 7, 7]);  getitem_687 = None
        empty_480 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_1 = torch.ops.aten.expand.default(empty_480, [512, 512, 7, 7]);  empty_480 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(view_1367, expand_1, convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_1 = convert_element_type_53 = None
        getitem_689 = convolution_backward[0];  convolution_backward = None
        empty_481 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_2 = torch.ops.aten.expand.default(empty_481, [2048, 512, 1, 1]);  empty_481 = None
        view_1368 = torch.ops.aten.view.default(getitem_688, [512, 49, 512]);  getitem_688 = None
        view_1369 = torch.ops.aten.view.default(view_1368, [512, 512, 7, 7]);  view_1368 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(view_1367, view_1369, expand_2, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1367 = view_1369 = expand_2 = None
        getitem_693 = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(getitem_693, torch.float32);  getitem_693 = None
        full_default_313 = torch.ops.aten.full.default([25088, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_317 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 628, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_665, 'Y_ptr': full_default_313, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_665 = None
        getitem_695 = triton_kernel_wrapper_functional_proxy_317['Y_ptr'];  triton_kernel_wrapper_functional_proxy_317 = None
        view_1372 = torch.ops.aten.view.default(getitem_695, [512, 49, 512]);  getitem_695 = None
        view_1373 = torch.ops.aten.view.default(view_1372, [512, 512, 7, 7]);  view_1372 = None
        mul_372 = torch.ops.aten.mul.Tensor(getitem_689, view_1373);  getitem_689 = view_1373 = None
        empty_482 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_318 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 440, constant_args_idx = 629, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_660, 'S_ptr': getitem_661, 'M_ptr': getitem_662, 'Y_ptr': empty_482, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_660 = getitem_661 = getitem_662 = empty_482 = None
        getitem_696 = triton_kernel_wrapper_functional_proxy_318['Y_ptr'];  triton_kernel_wrapper_functional_proxy_318 = None
        view_1388 = torch.ops.aten.view.default(mul_372, [512, 512, 49]);  mul_372 = None
        view_1389 = torch.ops.aten.view.default(getitem_696, [512, 49, 512]);  getitem_696 = None
        view_1390 = torch.ops.aten.view.default(view_1389, [512, 512, 49]);  view_1389 = None
        full_default_76 = torch.ops.aten.full.default([512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_319 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 441, constant_args_idx = 630, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1390, 'DY': view_1388, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_697 = triton_kernel_wrapper_functional_proxy_319['DBETA']
        getitem_698 = triton_kernel_wrapper_functional_proxy_319['DGAMMA'];  triton_kernel_wrapper_functional_proxy_319 = None
        empty_483 = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_158 = torch.ops.aten.permute.default(empty_483, [0, 1, 2]);  empty_483 = None
        triton_kernel_wrapper_functional_proxy_320 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 442, constant_args_idx = 631, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1390, 'DY': view_1388, 'INVSTD': rsqrt_51, 'GAMMA': primals_310, 'DBETA': getitem_697, 'DGAMMA': getitem_698, 'DX': permute_158, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1390 = view_1388 = rsqrt_51 = primals_310 = permute_158 = None
        getitem_699 = triton_kernel_wrapper_functional_proxy_320['DX'];  triton_kernel_wrapper_functional_proxy_320 = None
        convert_element_type_default_103 = torch.ops.prims.convert_element_type.default(getitem_698, torch.float32);  getitem_698 = None
        convert_element_type_default_102 = torch.ops.prims.convert_element_type.default(getitem_697, torch.float32);  getitem_697 = None
        empty_484 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_321 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 443, constant_args_idx = 632, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_653, 'S_ptr': getitem_654, 'M_ptr': getitem_655, 'Y_ptr': empty_484, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_653 = getitem_654 = getitem_655 = empty_484 = None
        getitem_700 = triton_kernel_wrapper_functional_proxy_321['Y_ptr'];  triton_kernel_wrapper_functional_proxy_321 = None
        view_1406 = torch.ops.aten.view.default(getitem_699, [512, 512, 7, 7]);  getitem_699 = None
        empty_485 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_3 = torch.ops.aten.expand.default(empty_485, [512, 512, 7, 7]);  empty_485 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_1406, expand_3, convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_3 = convert_element_type_52 = None
        getitem_701 = convolution_backward_2[0];  convolution_backward_2 = None
        empty_486 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_4 = torch.ops.aten.expand.default(empty_486, [512, 512, 3, 3]);  empty_486 = None
        view_1407 = torch.ops.aten.view.default(getitem_700, [512, 49, 512]);  getitem_700 = None
        view_1408 = torch.ops.aten.view.default(view_1407, [512, 512, 7, 7]);  view_1407 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_1406, view_1408, expand_4, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1406 = view_1408 = expand_4 = None
        getitem_705 = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_72 = torch.ops.prims.convert_element_type.default(getitem_705, torch.float32);  getitem_705 = None
        triton_kernel_wrapper_functional_proxy_322 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 633, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_652, 'Y_ptr': full_default_313, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_652 = None
        getitem_707 = triton_kernel_wrapper_functional_proxy_322['Y_ptr'];  triton_kernel_wrapper_functional_proxy_322 = None
        view_1411 = torch.ops.aten.view.default(getitem_707, [512, 49, 512]);  getitem_707 = None
        view_1412 = torch.ops.aten.view.default(view_1411, [512, 512, 7, 7]);  view_1411 = None
        mul_373 = torch.ops.aten.mul.Tensor(getitem_701, view_1412);  getitem_701 = view_1412 = None
        empty_487 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_323 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 444, constant_args_idx = 634, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_647, 'S_ptr': getitem_648, 'M_ptr': getitem_649, 'Y_ptr': empty_487, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_647 = getitem_648 = getitem_649 = empty_487 = None
        getitem_708 = triton_kernel_wrapper_functional_proxy_323['Y_ptr'];  triton_kernel_wrapper_functional_proxy_323 = None
        view_1427 = torch.ops.aten.view.default(mul_373, [512, 512, 49]);  mul_373 = None
        view_1428 = torch.ops.aten.view.default(getitem_708, [512, 49, 512]);  getitem_708 = None
        view_1429 = torch.ops.aten.view.default(view_1428, [512, 512, 49]);  view_1428 = None
        triton_kernel_wrapper_functional_proxy_324 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 445, constant_args_idx = 635, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1429, 'DY': view_1427, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_709 = triton_kernel_wrapper_functional_proxy_324['DBETA']
        getitem_710 = triton_kernel_wrapper_functional_proxy_324['DGAMMA'];  triton_kernel_wrapper_functional_proxy_324 = None
        empty_488 = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_159 = torch.ops.aten.permute.default(empty_488, [0, 1, 2]);  empty_488 = None
        triton_kernel_wrapper_functional_proxy_325 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 446, constant_args_idx = 636, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1429, 'DY': view_1427, 'INVSTD': rsqrt_50, 'GAMMA': primals_304, 'DBETA': getitem_709, 'DGAMMA': getitem_710, 'DX': permute_159, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1429 = view_1427 = rsqrt_50 = primals_304 = permute_159 = None
        getitem_711 = triton_kernel_wrapper_functional_proxy_325['DX'];  triton_kernel_wrapper_functional_proxy_325 = None
        convert_element_type_default_101 = torch.ops.prims.convert_element_type.default(getitem_710, torch.float32);  getitem_710 = None
        convert_element_type_default_100 = torch.ops.prims.convert_element_type.default(getitem_709, torch.float32);  getitem_709 = None
        empty_489 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_326 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 447, constant_args_idx = 637, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_640, 'S_ptr': getitem_641, 'M_ptr': getitem_642, 'Y_ptr': empty_489, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_640 = getitem_641 = getitem_642 = empty_489 = None
        getitem_712 = triton_kernel_wrapper_functional_proxy_326['Y_ptr'];  triton_kernel_wrapper_functional_proxy_326 = None
        view_1445 = torch.ops.aten.view.default(getitem_711, [512, 512, 7, 7]);  getitem_711 = None
        empty_490 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_5 = torch.ops.aten.expand.default(empty_490, [512, 2048, 7, 7]);  empty_490 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(view_1445, expand_5, convert_element_type_51, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_5 = convert_element_type_51 = None
        getitem_713 = convolution_backward_4[0];  convolution_backward_4 = None
        empty_491 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_6 = torch.ops.aten.expand.default(empty_491, [512, 2048, 1, 1]);  empty_491 = None
        view_1446 = torch.ops.aten.view.default(getitem_712, [512, 196, 512]);  getitem_712 = None
        view_1447 = torch.ops.aten.view.default(view_1446, [512, 2048, 7, 7]);  view_1446 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(view_1445, view_1447, expand_6, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1445 = view_1447 = expand_6 = None
        getitem_717 = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_77 = torch.ops.prims.convert_element_type.default(getitem_717, torch.float32);  getitem_717 = None
        add_228 = torch.ops.aten.add.Tensor(mul_371, getitem_713);  mul_371 = getitem_713 = None
        triton_kernel_wrapper_functional_proxy_327 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 638, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_639, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_639 = None
        getitem_719 = triton_kernel_wrapper_functional_proxy_327['Y_ptr'];  triton_kernel_wrapper_functional_proxy_327 = None
        view_1450 = torch.ops.aten.view.default(getitem_719, [512, 196, 512]);  getitem_719 = None
        view_1451 = torch.ops.aten.view.default(view_1450, [512, 2048, 7, 7]);  view_1450 = None
        mul_374 = torch.ops.aten.mul.Tensor(add_228, view_1451);  add_228 = view_1451 = None
        empty_492 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_328 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 448, constant_args_idx = 639, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_634, 'S_ptr': getitem_635, 'M_ptr': getitem_636, 'Y_ptr': empty_492, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_634 = getitem_635 = getitem_636 = empty_492 = None
        getitem_720 = triton_kernel_wrapper_functional_proxy_328['Y_ptr'];  triton_kernel_wrapper_functional_proxy_328 = None
        view_1466 = torch.ops.aten.view.default(mul_374, [512, 2048, 49])
        view_1467 = torch.ops.aten.view.default(getitem_720, [512, 196, 512]);  getitem_720 = None
        view_1468 = torch.ops.aten.view.default(view_1467, [512, 2048, 49]);  view_1467 = None
        triton_kernel_wrapper_functional_proxy_329 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 449, constant_args_idx = 640, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1468, 'DY': view_1466, 'DBETA': full_default_264, 'DGAMMA': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_721 = triton_kernel_wrapper_functional_proxy_329['DBETA']
        getitem_722 = triton_kernel_wrapper_functional_proxy_329['DGAMMA'];  triton_kernel_wrapper_functional_proxy_329 = None
        empty_493 = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_160 = torch.ops.aten.permute.default(empty_493, [0, 1, 2]);  empty_493 = None
        triton_kernel_wrapper_functional_proxy_330 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 450, constant_args_idx = 641, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1468, 'DY': view_1466, 'INVSTD': rsqrt_49, 'GAMMA': primals_298, 'DBETA': getitem_721, 'DGAMMA': getitem_722, 'DX': permute_160, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1468 = view_1466 = rsqrt_49 = primals_298 = permute_160 = None
        getitem_723 = triton_kernel_wrapper_functional_proxy_330['DX'];  triton_kernel_wrapper_functional_proxy_330 = None
        convert_element_type_default_99 = torch.ops.prims.convert_element_type.default(getitem_722, torch.float32);  getitem_722 = None
        convert_element_type_default_98 = torch.ops.prims.convert_element_type.default(getitem_721, torch.float32);  getitem_721 = None
        empty_494 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_331 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 451, constant_args_idx = 642, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_627, 'S_ptr': getitem_628, 'M_ptr': getitem_629, 'Y_ptr': empty_494, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_627 = getitem_628 = getitem_629 = empty_494 = None
        getitem_724 = triton_kernel_wrapper_functional_proxy_331['Y_ptr'];  triton_kernel_wrapper_functional_proxy_331 = None
        view_1484 = torch.ops.aten.view.default(getitem_723, [512, 2048, 7, 7]);  getitem_723 = None
        empty_495 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_7 = torch.ops.aten.expand.default(empty_495, [512, 512, 7, 7]);  empty_495 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(view_1484, expand_7, convert_element_type_50, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_7 = convert_element_type_50 = None
        getitem_725 = convolution_backward_6[0];  convolution_backward_6 = None
        empty_496 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_8 = torch.ops.aten.expand.default(empty_496, [2048, 512, 1, 1]);  empty_496 = None
        view_1485 = torch.ops.aten.view.default(getitem_724, [512, 49, 512]);  getitem_724 = None
        view_1486 = torch.ops.aten.view.default(view_1485, [512, 512, 7, 7]);  view_1485 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_1484, view_1486, expand_8, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1484 = view_1486 = expand_8 = None
        getitem_729 = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_82 = torch.ops.prims.convert_element_type.default(getitem_729, torch.float32);  getitem_729 = None
        triton_kernel_wrapper_functional_proxy_332 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 643, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_626, 'Y_ptr': full_default_313, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_626 = None
        getitem_731 = triton_kernel_wrapper_functional_proxy_332['Y_ptr'];  triton_kernel_wrapper_functional_proxy_332 = None
        view_1489 = torch.ops.aten.view.default(getitem_731, [512, 49, 512]);  getitem_731 = None
        view_1490 = torch.ops.aten.view.default(view_1489, [512, 512, 7, 7]);  view_1489 = None
        mul_375 = torch.ops.aten.mul.Tensor(getitem_725, view_1490);  getitem_725 = view_1490 = None
        empty_497 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_333 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 452, constant_args_idx = 644, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_621, 'S_ptr': getitem_622, 'M_ptr': getitem_623, 'Y_ptr': empty_497, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_621 = getitem_622 = getitem_623 = empty_497 = None
        getitem_732 = triton_kernel_wrapper_functional_proxy_333['Y_ptr'];  triton_kernel_wrapper_functional_proxy_333 = None
        view_1505 = torch.ops.aten.view.default(mul_375, [512, 512, 49]);  mul_375 = None
        view_1506 = torch.ops.aten.view.default(getitem_732, [512, 49, 512]);  getitem_732 = None
        view_1507 = torch.ops.aten.view.default(view_1506, [512, 512, 49]);  view_1506 = None
        triton_kernel_wrapper_functional_proxy_334 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 453, constant_args_idx = 645, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1507, 'DY': view_1505, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_733 = triton_kernel_wrapper_functional_proxy_334['DBETA']
        getitem_734 = triton_kernel_wrapper_functional_proxy_334['DGAMMA'];  triton_kernel_wrapper_functional_proxy_334 = None
        empty_498 = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_161 = torch.ops.aten.permute.default(empty_498, [0, 1, 2]);  empty_498 = None
        triton_kernel_wrapper_functional_proxy_335 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 454, constant_args_idx = 646, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1507, 'DY': view_1505, 'INVSTD': rsqrt_48, 'GAMMA': primals_292, 'DBETA': getitem_733, 'DGAMMA': getitem_734, 'DX': permute_161, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1507 = view_1505 = rsqrt_48 = primals_292 = permute_161 = None
        getitem_735 = triton_kernel_wrapper_functional_proxy_335['DX'];  triton_kernel_wrapper_functional_proxy_335 = None
        convert_element_type_default_97 = torch.ops.prims.convert_element_type.default(getitem_734, torch.float32);  getitem_734 = None
        convert_element_type_default_96 = torch.ops.prims.convert_element_type.default(getitem_733, torch.float32);  getitem_733 = None
        empty_499 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_336 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 455, constant_args_idx = 647, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_614, 'S_ptr': getitem_615, 'M_ptr': getitem_616, 'Y_ptr': empty_499, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_614 = getitem_615 = getitem_616 = empty_499 = None
        getitem_736 = triton_kernel_wrapper_functional_proxy_336['Y_ptr'];  triton_kernel_wrapper_functional_proxy_336 = None
        view_1523 = torch.ops.aten.view.default(getitem_735, [512, 512, 7, 7]);  getitem_735 = None
        empty_500 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_9 = torch.ops.aten.expand.default(empty_500, [512, 512, 7, 7]);  empty_500 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_1523, expand_9, convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_9 = convert_element_type_49 = None
        getitem_737 = convolution_backward_8[0];  convolution_backward_8 = None
        empty_501 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_10 = torch.ops.aten.expand.default(empty_501, [512, 512, 3, 3]);  empty_501 = None
        view_1524 = torch.ops.aten.view.default(getitem_736, [512, 49, 512]);  getitem_736 = None
        view_1525 = torch.ops.aten.view.default(view_1524, [512, 512, 7, 7]);  view_1524 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(view_1523, view_1525, expand_10, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1523 = view_1525 = expand_10 = None
        getitem_741 = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(getitem_741, torch.float32);  getitem_741 = None
        triton_kernel_wrapper_functional_proxy_337 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 648, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_613, 'Y_ptr': full_default_313, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_613 = None
        getitem_743 = triton_kernel_wrapper_functional_proxy_337['Y_ptr'];  triton_kernel_wrapper_functional_proxy_337 = None
        view_1528 = torch.ops.aten.view.default(getitem_743, [512, 49, 512]);  getitem_743 = None
        view_1529 = torch.ops.aten.view.default(view_1528, [512, 512, 7, 7]);  view_1528 = None
        mul_376 = torch.ops.aten.mul.Tensor(getitem_737, view_1529);  getitem_737 = view_1529 = None
        empty_502 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_338 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 456, constant_args_idx = 649, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_608, 'S_ptr': getitem_609, 'M_ptr': getitem_610, 'Y_ptr': empty_502, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_608 = getitem_609 = getitem_610 = empty_502 = None
        getitem_744 = triton_kernel_wrapper_functional_proxy_338['Y_ptr'];  triton_kernel_wrapper_functional_proxy_338 = None
        view_1544 = torch.ops.aten.view.default(mul_376, [512, 512, 49]);  mul_376 = None
        view_1545 = torch.ops.aten.view.default(getitem_744, [512, 49, 512]);  getitem_744 = None
        view_1546 = torch.ops.aten.view.default(view_1545, [512, 512, 49]);  view_1545 = None
        triton_kernel_wrapper_functional_proxy_339 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 457, constant_args_idx = 650, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1546, 'DY': view_1544, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_745 = triton_kernel_wrapper_functional_proxy_339['DBETA']
        getitem_746 = triton_kernel_wrapper_functional_proxy_339['DGAMMA'];  triton_kernel_wrapper_functional_proxy_339 = None
        empty_503 = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_162 = torch.ops.aten.permute.default(empty_503, [0, 1, 2]);  empty_503 = None
        triton_kernel_wrapper_functional_proxy_340 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 458, constant_args_idx = 651, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1546, 'DY': view_1544, 'INVSTD': rsqrt_47, 'GAMMA': primals_286, 'DBETA': getitem_745, 'DGAMMA': getitem_746, 'DX': permute_162, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1546 = view_1544 = rsqrt_47 = primals_286 = permute_162 = None
        getitem_747 = triton_kernel_wrapper_functional_proxy_340['DX'];  triton_kernel_wrapper_functional_proxy_340 = None
        convert_element_type_default_95 = torch.ops.prims.convert_element_type.default(getitem_746, torch.float32);  getitem_746 = None
        convert_element_type_default_94 = torch.ops.prims.convert_element_type.default(getitem_745, torch.float32);  getitem_745 = None
        empty_504 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_341 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 459, constant_args_idx = 652, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_601, 'S_ptr': getitem_602, 'M_ptr': getitem_603, 'Y_ptr': empty_504, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_601 = getitem_602 = getitem_603 = empty_504 = None
        getitem_748 = triton_kernel_wrapper_functional_proxy_341['Y_ptr'];  triton_kernel_wrapper_functional_proxy_341 = None
        view_1562 = torch.ops.aten.view.default(getitem_747, [512, 512, 7, 7]);  getitem_747 = None
        empty_505 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_11 = torch.ops.aten.expand.default(empty_505, [512, 2048, 7, 7]);  empty_505 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_1562, expand_11, convert_element_type_48, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_11 = convert_element_type_48 = None
        getitem_749 = convolution_backward_10[0];  convolution_backward_10 = None
        empty_506 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_12 = torch.ops.aten.expand.default(empty_506, [512, 2048, 1, 1]);  empty_506 = None
        view_1563 = torch.ops.aten.view.default(getitem_748, [512, 196, 512]);  getitem_748 = None
        view_1564 = torch.ops.aten.view.default(view_1563, [512, 2048, 7, 7]);  view_1563 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(view_1562, view_1564, expand_12, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1562 = view_1564 = expand_12 = None
        getitem_753 = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(getitem_753, torch.float32);  getitem_753 = None
        add_229 = torch.ops.aten.add.Tensor(mul_374, getitem_749);  mul_374 = getitem_749 = None
        triton_kernel_wrapper_functional_proxy_342 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 653, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_600, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_600 = None
        getitem_755 = triton_kernel_wrapper_functional_proxy_342['Y_ptr'];  triton_kernel_wrapper_functional_proxy_342 = None
        view_1567 = torch.ops.aten.view.default(getitem_755, [512, 196, 512]);  getitem_755 = None
        view_1568 = torch.ops.aten.view.default(view_1567, [512, 2048, 7, 7]);  view_1567 = None
        mul_377 = torch.ops.aten.mul.Tensor(add_229, view_1568);  add_229 = view_1568 = None
        empty_507 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_343 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 460, constant_args_idx = 654, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_595, 'S_ptr': getitem_596, 'M_ptr': getitem_597, 'Y_ptr': empty_507, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_595 = getitem_596 = getitem_597 = empty_507 = None
        getitem_756 = triton_kernel_wrapper_functional_proxy_343['Y_ptr'];  triton_kernel_wrapper_functional_proxy_343 = None
        view_1583 = torch.ops.aten.view.default(mul_377, [512, 2048, 49]);  mul_377 = None
        view_1584 = torch.ops.aten.view.default(getitem_756, [512, 196, 512]);  getitem_756 = None
        view_1585 = torch.ops.aten.view.default(view_1584, [512, 2048, 49]);  view_1584 = None
        triton_kernel_wrapper_functional_proxy_344 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 461, constant_args_idx = 655, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1585, 'DY': view_1583, 'DBETA': full_default_264, 'DGAMMA': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_757 = triton_kernel_wrapper_functional_proxy_344['DBETA']
        getitem_758 = triton_kernel_wrapper_functional_proxy_344['DGAMMA'];  triton_kernel_wrapper_functional_proxy_344 = None
        empty_508 = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_163 = torch.ops.aten.permute.default(empty_508, [0, 1, 2]);  empty_508 = None
        triton_kernel_wrapper_functional_proxy_345 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 462, constant_args_idx = 656, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1585, 'DY': view_1583, 'INVSTD': rsqrt_46, 'GAMMA': primals_280, 'DBETA': getitem_757, 'DGAMMA': getitem_758, 'DX': permute_163, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1585 = rsqrt_46 = primals_280 = permute_163 = None
        getitem_759 = triton_kernel_wrapper_functional_proxy_345['DX'];  triton_kernel_wrapper_functional_proxy_345 = None
        convert_element_type_default_93 = torch.ops.prims.convert_element_type.default(getitem_758, torch.float32);  getitem_758 = None
        convert_element_type_default_92 = torch.ops.prims.convert_element_type.default(getitem_757, torch.float32);  getitem_757 = None
        empty_509 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_346 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 463, constant_args_idx = 657, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_588, 'S_ptr': getitem_589, 'M_ptr': getitem_590, 'Y_ptr': empty_509, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_588 = getitem_589 = getitem_590 = empty_509 = None
        getitem_760 = triton_kernel_wrapper_functional_proxy_346['Y_ptr'];  triton_kernel_wrapper_functional_proxy_346 = None
        view_1601 = torch.ops.aten.view.default(getitem_759, [512, 2048, 7, 7]);  getitem_759 = None
        empty_510 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_13 = torch.ops.aten.expand.default(empty_510, [512, 1024, 14, 14]);  empty_510 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_1601, expand_13, convert_element_type_47, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_13 = convert_element_type_47 = None
        getitem_761 = convolution_backward_12[0];  convolution_backward_12 = None
        empty_511 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_14 = torch.ops.aten.expand.default(empty_511, [2048, 1024, 1, 1]);  empty_511 = None
        view_1602 = torch.ops.aten.view.default(getitem_760, [512, 392, 512]);  getitem_760 = None
        view_1603 = torch.ops.aten.view.default(view_1602, [512, 1024, 14, 14]);  view_1602 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(view_1601, view_1603, expand_14, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1601 = view_1603 = expand_14 = None
        getitem_765 = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(getitem_765, torch.float32);  getitem_765 = None
        empty_512 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_347 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 464, constant_args_idx = 658, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_585, 'S_ptr': getitem_586, 'M_ptr': getitem_587, 'Y_ptr': empty_512, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_585 = getitem_586 = getitem_587 = empty_512 = None
        getitem_767 = triton_kernel_wrapper_functional_proxy_347['Y_ptr'];  triton_kernel_wrapper_functional_proxy_347 = None
        view_1619 = torch.ops.aten.view.default(getitem_767, [512, 196, 512]);  getitem_767 = None
        view_1620 = torch.ops.aten.view.default(view_1619, [512, 2048, 49]);  view_1619 = None
        triton_kernel_wrapper_functional_proxy_348 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 465, constant_args_idx = 659, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1620, 'DY': view_1583, 'DBETA': full_default_264, 'DGAMMA': full_default_264, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_264 = None
        getitem_768 = triton_kernel_wrapper_functional_proxy_348['DBETA']
        getitem_769 = triton_kernel_wrapper_functional_proxy_348['DGAMMA'];  triton_kernel_wrapper_functional_proxy_348 = None
        empty_513 = torch.ops.aten.empty.memory_format([512, 2048, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_164 = torch.ops.aten.permute.default(empty_513, [0, 1, 2]);  empty_513 = None
        triton_kernel_wrapper_functional_proxy_349 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 466, constant_args_idx = 660, grid = [(2048, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1620, 'DY': view_1583, 'INVSTD': rsqrt_45, 'GAMMA': primals_274, 'DBETA': getitem_768, 'DGAMMA': getitem_769, 'DX': permute_164, 'M': 25088, 'HW': 49, 'stride_n': 100352, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1620 = view_1583 = rsqrt_45 = primals_274 = permute_164 = None
        getitem_770 = triton_kernel_wrapper_functional_proxy_349['DX'];  triton_kernel_wrapper_functional_proxy_349 = None
        convert_element_type_default_91 = torch.ops.prims.convert_element_type.default(getitem_769, torch.float32);  getitem_769 = None
        convert_element_type_default_90 = torch.ops.prims.convert_element_type.default(getitem_768, torch.float32);  getitem_768 = None
        empty_514 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_350 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 467, constant_args_idx = 661, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_578, 'S_ptr': getitem_579, 'M_ptr': getitem_580, 'Y_ptr': empty_514, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_578 = getitem_579 = getitem_580 = empty_514 = None
        getitem_771 = triton_kernel_wrapper_functional_proxy_350['Y_ptr'];  triton_kernel_wrapper_functional_proxy_350 = None
        view_1636 = torch.ops.aten.view.default(getitem_770, [512, 2048, 7, 7]);  getitem_770 = None
        empty_515 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_15 = torch.ops.aten.expand.default(empty_515, [512, 512, 7, 7]);  empty_515 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(view_1636, expand_15, convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_15 = convert_element_type_46 = None
        getitem_772 = convolution_backward_14[0];  convolution_backward_14 = None
        empty_516 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_16 = torch.ops.aten.expand.default(empty_516, [2048, 512, 1, 1]);  empty_516 = None
        view_1637 = torch.ops.aten.view.default(getitem_771, [512, 49, 512]);  getitem_771 = None
        view_1638 = torch.ops.aten.view.default(view_1637, [512, 512, 7, 7]);  view_1637 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(view_1636, view_1638, expand_16, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1636 = view_1638 = expand_16 = None
        getitem_776 = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_102 = torch.ops.prims.convert_element_type.default(getitem_776, torch.float32);  getitem_776 = None
        triton_kernel_wrapper_functional_proxy_351 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 662, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_577, 'Y_ptr': full_default_313, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_577 = full_default_313 = None
        getitem_778 = triton_kernel_wrapper_functional_proxy_351['Y_ptr'];  triton_kernel_wrapper_functional_proxy_351 = None
        view_1641 = torch.ops.aten.view.default(getitem_778, [512, 49, 512]);  getitem_778 = None
        view_1642 = torch.ops.aten.view.default(view_1641, [512, 512, 7, 7]);  view_1641 = None
        mul_378 = torch.ops.aten.mul.Tensor(getitem_772, view_1642);  getitem_772 = view_1642 = None
        empty_517 = torch.ops.aten.empty.memory_format([25088, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_352 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 468, constant_args_idx = 663, grid = [(25088, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_572, 'S_ptr': getitem_573, 'M_ptr': getitem_574, 'Y_ptr': empty_517, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_572 = getitem_573 = getitem_574 = empty_517 = None
        getitem_779 = triton_kernel_wrapper_functional_proxy_352['Y_ptr'];  triton_kernel_wrapper_functional_proxy_352 = None
        view_1657 = torch.ops.aten.view.default(mul_378, [512, 512, 49]);  mul_378 = None
        view_1658 = torch.ops.aten.view.default(getitem_779, [512, 49, 512]);  getitem_779 = None
        view_1659 = torch.ops.aten.view.default(view_1658, [512, 512, 49]);  view_1658 = None
        triton_kernel_wrapper_functional_proxy_353 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 469, constant_args_idx = 664, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1659, 'DY': view_1657, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_780 = triton_kernel_wrapper_functional_proxy_353['DBETA']
        getitem_781 = triton_kernel_wrapper_functional_proxy_353['DGAMMA'];  triton_kernel_wrapper_functional_proxy_353 = None
        empty_518 = torch.ops.aten.empty.memory_format([512, 512, 49], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_165 = torch.ops.aten.permute.default(empty_518, [0, 1, 2]);  empty_518 = None
        triton_kernel_wrapper_functional_proxy_354 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 470, constant_args_idx = 665, grid = [(512, 25, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1659, 'DY': view_1657, 'INVSTD': rsqrt_44, 'GAMMA': primals_268, 'DBETA': getitem_780, 'DGAMMA': getitem_781, 'DX': permute_165, 'M': 25088, 'HW': 49, 'stride_n': 25088, 'stride_c': 49, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1659 = view_1657 = rsqrt_44 = primals_268 = permute_165 = None
        getitem_782 = triton_kernel_wrapper_functional_proxy_354['DX'];  triton_kernel_wrapper_functional_proxy_354 = None
        convert_element_type_default_89 = torch.ops.prims.convert_element_type.default(getitem_781, torch.float32);  getitem_781 = None
        convert_element_type_default_88 = torch.ops.prims.convert_element_type.default(getitem_780, torch.float32);  getitem_780 = None
        empty_519 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_355 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 471, constant_args_idx = 666, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_565, 'S_ptr': getitem_566, 'M_ptr': getitem_567, 'Y_ptr': empty_519, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_565 = getitem_566 = getitem_567 = empty_519 = None
        getitem_783 = triton_kernel_wrapper_functional_proxy_355['Y_ptr'];  triton_kernel_wrapper_functional_proxy_355 = None
        view_1675 = torch.ops.aten.view.default(getitem_782, [512, 512, 7, 7]);  getitem_782 = None
        empty_520 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_17 = torch.ops.aten.expand.default(empty_520, [512, 512, 14, 14]);  empty_520 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(view_1675, expand_17, convert_element_type_45, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_17 = convert_element_type_45 = None
        getitem_784 = convolution_backward_16[0];  convolution_backward_16 = None
        empty_521 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_18 = torch.ops.aten.expand.default(empty_521, [512, 512, 3, 3]);  empty_521 = None
        view_1676 = torch.ops.aten.view.default(getitem_783, [512, 196, 512]);  getitem_783 = None
        view_1677 = torch.ops.aten.view.default(view_1676, [512, 512, 14, 14]);  view_1676 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_1675, view_1677, expand_18, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1675 = view_1677 = expand_18 = None
        getitem_788 = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_107 = torch.ops.prims.convert_element_type.default(getitem_788, torch.float32);  getitem_788 = None
        triton_kernel_wrapper_functional_proxy_356 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 667, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_564, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_564 = None
        getitem_790 = triton_kernel_wrapper_functional_proxy_356['Y_ptr'];  triton_kernel_wrapper_functional_proxy_356 = None
        view_1680 = torch.ops.aten.view.default(getitem_790, [512, 196, 512]);  getitem_790 = None
        view_1681 = torch.ops.aten.view.default(view_1680, [512, 512, 14, 14]);  view_1680 = None
        mul_379 = torch.ops.aten.mul.Tensor(getitem_784, view_1681);  getitem_784 = view_1681 = None
        empty_522 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_357 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 472, constant_args_idx = 668, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_559, 'S_ptr': getitem_560, 'M_ptr': getitem_561, 'Y_ptr': empty_522, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_559 = getitem_560 = getitem_561 = empty_522 = None
        getitem_791 = triton_kernel_wrapper_functional_proxy_357['Y_ptr'];  triton_kernel_wrapper_functional_proxy_357 = None
        view_1696 = torch.ops.aten.view.default(mul_379, [512, 512, 196]);  mul_379 = None
        view_1697 = torch.ops.aten.view.default(getitem_791, [512, 196, 512]);  getitem_791 = None
        view_1698 = torch.ops.aten.view.default(view_1697, [512, 512, 196]);  view_1697 = None
        triton_kernel_wrapper_functional_proxy_358 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 473, constant_args_idx = 669, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1698, 'DY': view_1696, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_792 = triton_kernel_wrapper_functional_proxy_358['DBETA']
        getitem_793 = triton_kernel_wrapper_functional_proxy_358['DGAMMA'];  triton_kernel_wrapper_functional_proxy_358 = None
        empty_523 = torch.ops.aten.empty.memory_format([512, 512, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_166 = torch.ops.aten.permute.default(empty_523, [0, 1, 2]);  empty_523 = None
        triton_kernel_wrapper_functional_proxy_359 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 474, constant_args_idx = 670, grid = [(512, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1698, 'DY': view_1696, 'INVSTD': rsqrt_43, 'GAMMA': primals_262, 'DBETA': getitem_792, 'DGAMMA': getitem_793, 'DX': permute_166, 'M': 100352, 'HW': 196, 'stride_n': 100352, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1698 = view_1696 = rsqrt_43 = primals_262 = permute_166 = None
        getitem_794 = triton_kernel_wrapper_functional_proxy_359['DX'];  triton_kernel_wrapper_functional_proxy_359 = None
        convert_element_type_default_87 = torch.ops.prims.convert_element_type.default(getitem_793, torch.float32);  getitem_793 = None
        convert_element_type_default_86 = torch.ops.prims.convert_element_type.default(getitem_792, torch.float32);  getitem_792 = None
        empty_524 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_360 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 475, constant_args_idx = 671, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_552, 'S_ptr': getitem_553, 'M_ptr': getitem_554, 'Y_ptr': empty_524, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_552 = getitem_553 = getitem_554 = empty_524 = None
        getitem_795 = triton_kernel_wrapper_functional_proxy_360['Y_ptr'];  triton_kernel_wrapper_functional_proxy_360 = None
        view_1714 = torch.ops.aten.view.default(getitem_794, [512, 512, 14, 14]);  getitem_794 = None
        empty_525 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_19 = torch.ops.aten.expand.default(empty_525, [512, 1024, 14, 14]);  empty_525 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(view_1714, expand_19, convert_element_type_44, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_19 = convert_element_type_44 = None
        getitem_796 = convolution_backward_18[0];  convolution_backward_18 = None
        empty_526 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_20 = torch.ops.aten.expand.default(empty_526, [512, 1024, 1, 1]);  empty_526 = None
        view_1715 = torch.ops.aten.view.default(getitem_795, [512, 392, 512]);  getitem_795 = None
        view_1716 = torch.ops.aten.view.default(view_1715, [512, 1024, 14, 14]);  view_1715 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(view_1714, view_1716, expand_20, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1714 = view_1716 = expand_20 = None
        getitem_800 = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_112 = torch.ops.prims.convert_element_type.default(getitem_800, torch.float32);  getitem_800 = None
        add_230 = torch.ops.aten.add.Tensor(getitem_761, getitem_796);  getitem_761 = getitem_796 = None
        full_default_339 = torch.ops.aten.full.default([200704, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_361 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 672, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_551, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_551 = None
        getitem_802 = triton_kernel_wrapper_functional_proxy_361['Y_ptr'];  triton_kernel_wrapper_functional_proxy_361 = None
        view_1719 = torch.ops.aten.view.default(getitem_802, [512, 392, 512]);  getitem_802 = None
        view_1720 = torch.ops.aten.view.default(view_1719, [512, 1024, 14, 14]);  view_1719 = None
        mul_380 = torch.ops.aten.mul.Tensor(add_230, view_1720);  add_230 = view_1720 = None
        empty_527 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_362 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 476, constant_args_idx = 673, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_546, 'S_ptr': getitem_547, 'M_ptr': getitem_548, 'Y_ptr': empty_527, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_546 = getitem_547 = getitem_548 = empty_527 = None
        getitem_803 = triton_kernel_wrapper_functional_proxy_362['Y_ptr'];  triton_kernel_wrapper_functional_proxy_362 = None
        view_1735 = torch.ops.aten.view.default(mul_380, [512, 1024, 196])
        view_1736 = torch.ops.aten.view.default(getitem_803, [512, 392, 512]);  getitem_803 = None
        view_1737 = torch.ops.aten.view.default(view_1736, [512, 1024, 196]);  view_1736 = None
        full_default_152 = torch.ops.aten.full.default([1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_363 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 477, constant_args_idx = 674, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1737, 'DY': view_1735, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_804 = triton_kernel_wrapper_functional_proxy_363['DBETA']
        getitem_805 = triton_kernel_wrapper_functional_proxy_363['DGAMMA'];  triton_kernel_wrapper_functional_proxy_363 = None
        empty_528 = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_167 = torch.ops.aten.permute.default(empty_528, [0, 1, 2]);  empty_528 = None
        triton_kernel_wrapper_functional_proxy_364 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 478, constant_args_idx = 675, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1737, 'DY': view_1735, 'INVSTD': rsqrt_42, 'GAMMA': primals_256, 'DBETA': getitem_804, 'DGAMMA': getitem_805, 'DX': permute_167, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1737 = view_1735 = rsqrt_42 = primals_256 = permute_167 = None
        getitem_806 = triton_kernel_wrapper_functional_proxy_364['DX'];  triton_kernel_wrapper_functional_proxy_364 = None
        convert_element_type_default_85 = torch.ops.prims.convert_element_type.default(getitem_805, torch.float32);  getitem_805 = None
        convert_element_type_default_84 = torch.ops.prims.convert_element_type.default(getitem_804, torch.float32);  getitem_804 = None
        empty_529 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_365 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 479, constant_args_idx = 676, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_539, 'S_ptr': getitem_540, 'M_ptr': getitem_541, 'Y_ptr': empty_529, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_539 = getitem_540 = getitem_541 = empty_529 = None
        getitem_807 = triton_kernel_wrapper_functional_proxy_365['Y_ptr'];  triton_kernel_wrapper_functional_proxy_365 = None
        view_1753 = torch.ops.aten.view.default(getitem_806, [512, 1024, 14, 14]);  getitem_806 = None
        empty_530 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_21 = torch.ops.aten.expand.default(empty_530, [512, 256, 14, 14]);  empty_530 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_1753, expand_21, convert_element_type_43, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_21 = convert_element_type_43 = None
        getitem_808 = convolution_backward_20[0];  convolution_backward_20 = None
        empty_531 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_22 = torch.ops.aten.expand.default(empty_531, [1024, 256, 1, 1]);  empty_531 = None
        view_1754 = torch.ops.aten.view.default(getitem_807, [512, 98, 512]);  getitem_807 = None
        view_1755 = torch.ops.aten.view.default(view_1754, [512, 256, 14, 14]);  view_1754 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(view_1753, view_1755, expand_22, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1753 = view_1755 = expand_22 = None
        getitem_812 = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_117 = torch.ops.prims.convert_element_type.default(getitem_812, torch.float32);  getitem_812 = None
        full_default_342 = torch.ops.aten.full.default([50176, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_366 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 677, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_538, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_538 = None
        getitem_814 = triton_kernel_wrapper_functional_proxy_366['Y_ptr'];  triton_kernel_wrapper_functional_proxy_366 = None
        view_1758 = torch.ops.aten.view.default(getitem_814, [512, 98, 512]);  getitem_814 = None
        view_1759 = torch.ops.aten.view.default(view_1758, [512, 256, 14, 14]);  view_1758 = None
        mul_381 = torch.ops.aten.mul.Tensor(getitem_808, view_1759);  getitem_808 = view_1759 = None
        empty_532 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_367 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 480, constant_args_idx = 678, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_533, 'S_ptr': getitem_534, 'M_ptr': getitem_535, 'Y_ptr': empty_532, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_533 = getitem_534 = getitem_535 = empty_532 = None
        getitem_815 = triton_kernel_wrapper_functional_proxy_367['Y_ptr'];  triton_kernel_wrapper_functional_proxy_367 = None
        view_1774 = torch.ops.aten.view.default(mul_381, [512, 256, 196]);  mul_381 = None
        view_1775 = torch.ops.aten.view.default(getitem_815, [512, 98, 512]);  getitem_815 = None
        view_1776 = torch.ops.aten.view.default(view_1775, [512, 256, 196]);  view_1775 = None
        full_default_18 = torch.ops.aten.full.default([256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_368 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 481, constant_args_idx = 679, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1776, 'DY': view_1774, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_816 = triton_kernel_wrapper_functional_proxy_368['DBETA']
        getitem_817 = triton_kernel_wrapper_functional_proxy_368['DGAMMA'];  triton_kernel_wrapper_functional_proxy_368 = None
        empty_533 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_168 = torch.ops.aten.permute.default(empty_533, [0, 1, 2]);  empty_533 = None
        triton_kernel_wrapper_functional_proxy_369 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 482, constant_args_idx = 680, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1776, 'DY': view_1774, 'INVSTD': rsqrt_41, 'GAMMA': primals_250, 'DBETA': getitem_816, 'DGAMMA': getitem_817, 'DX': permute_168, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1776 = view_1774 = rsqrt_41 = primals_250 = permute_168 = None
        getitem_818 = triton_kernel_wrapper_functional_proxy_369['DX'];  triton_kernel_wrapper_functional_proxy_369 = None
        convert_element_type_default_83 = torch.ops.prims.convert_element_type.default(getitem_817, torch.float32);  getitem_817 = None
        convert_element_type_default_82 = torch.ops.prims.convert_element_type.default(getitem_816, torch.float32);  getitem_816 = None
        empty_534 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_370 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 483, constant_args_idx = 681, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_526, 'S_ptr': getitem_527, 'M_ptr': getitem_528, 'Y_ptr': empty_534, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_526 = getitem_527 = getitem_528 = empty_534 = None
        getitem_819 = triton_kernel_wrapper_functional_proxy_370['Y_ptr'];  triton_kernel_wrapper_functional_proxy_370 = None
        view_1792 = torch.ops.aten.view.default(getitem_818, [512, 256, 14, 14]);  getitem_818 = None
        empty_535 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_23 = torch.ops.aten.expand.default(empty_535, [512, 256, 14, 14]);  empty_535 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(view_1792, expand_23, convert_element_type_42, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_23 = convert_element_type_42 = None
        getitem_820 = convolution_backward_22[0];  convolution_backward_22 = None
        empty_536 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_24 = torch.ops.aten.expand.default(empty_536, [256, 256, 3, 3]);  empty_536 = None
        view_1793 = torch.ops.aten.view.default(getitem_819, [512, 98, 512]);  getitem_819 = None
        view_1794 = torch.ops.aten.view.default(view_1793, [512, 256, 14, 14]);  view_1793 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(view_1792, view_1794, expand_24, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1792 = view_1794 = expand_24 = None
        getitem_824 = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(getitem_824, torch.float32);  getitem_824 = None
        triton_kernel_wrapper_functional_proxy_371 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 682, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_525, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_525 = None
        getitem_826 = triton_kernel_wrapper_functional_proxy_371['Y_ptr'];  triton_kernel_wrapper_functional_proxy_371 = None
        view_1797 = torch.ops.aten.view.default(getitem_826, [512, 98, 512]);  getitem_826 = None
        view_1798 = torch.ops.aten.view.default(view_1797, [512, 256, 14, 14]);  view_1797 = None
        mul_382 = torch.ops.aten.mul.Tensor(getitem_820, view_1798);  getitem_820 = view_1798 = None
        empty_537 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_372 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 484, constant_args_idx = 683, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_520, 'S_ptr': getitem_521, 'M_ptr': getitem_522, 'Y_ptr': empty_537, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_520 = getitem_521 = getitem_522 = empty_537 = None
        getitem_827 = triton_kernel_wrapper_functional_proxy_372['Y_ptr'];  triton_kernel_wrapper_functional_proxy_372 = None
        view_1813 = torch.ops.aten.view.default(mul_382, [512, 256, 196]);  mul_382 = None
        view_1814 = torch.ops.aten.view.default(getitem_827, [512, 98, 512]);  getitem_827 = None
        view_1815 = torch.ops.aten.view.default(view_1814, [512, 256, 196]);  view_1814 = None
        triton_kernel_wrapper_functional_proxy_373 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 485, constant_args_idx = 684, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1815, 'DY': view_1813, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_828 = triton_kernel_wrapper_functional_proxy_373['DBETA']
        getitem_829 = triton_kernel_wrapper_functional_proxy_373['DGAMMA'];  triton_kernel_wrapper_functional_proxy_373 = None
        empty_538 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_169 = torch.ops.aten.permute.default(empty_538, [0, 1, 2]);  empty_538 = None
        triton_kernel_wrapper_functional_proxy_374 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 486, constant_args_idx = 685, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1815, 'DY': view_1813, 'INVSTD': rsqrt_40, 'GAMMA': primals_244, 'DBETA': getitem_828, 'DGAMMA': getitem_829, 'DX': permute_169, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1815 = view_1813 = rsqrt_40 = primals_244 = permute_169 = None
        getitem_830 = triton_kernel_wrapper_functional_proxy_374['DX'];  triton_kernel_wrapper_functional_proxy_374 = None
        convert_element_type_default_81 = torch.ops.prims.convert_element_type.default(getitem_829, torch.float32);  getitem_829 = None
        convert_element_type_default_80 = torch.ops.prims.convert_element_type.default(getitem_828, torch.float32);  getitem_828 = None
        empty_539 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_375 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 487, constant_args_idx = 686, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_513, 'S_ptr': getitem_514, 'M_ptr': getitem_515, 'Y_ptr': empty_539, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_513 = getitem_514 = getitem_515 = empty_539 = None
        getitem_831 = triton_kernel_wrapper_functional_proxy_375['Y_ptr'];  triton_kernel_wrapper_functional_proxy_375 = None
        view_1831 = torch.ops.aten.view.default(getitem_830, [512, 256, 14, 14]);  getitem_830 = None
        empty_540 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_25 = torch.ops.aten.expand.default(empty_540, [512, 1024, 14, 14]);  empty_540 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(view_1831, expand_25, convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_25 = convert_element_type_41 = None
        getitem_832 = convolution_backward_24[0];  convolution_backward_24 = None
        empty_541 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_26 = torch.ops.aten.expand.default(empty_541, [256, 1024, 1, 1]);  empty_541 = None
        view_1832 = torch.ops.aten.view.default(getitem_831, [512, 392, 512]);  getitem_831 = None
        view_1833 = torch.ops.aten.view.default(view_1832, [512, 1024, 14, 14]);  view_1832 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(view_1831, view_1833, expand_26, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1831 = view_1833 = expand_26 = None
        getitem_836 = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_127 = torch.ops.prims.convert_element_type.default(getitem_836, torch.float32);  getitem_836 = None
        add_231 = torch.ops.aten.add.Tensor(mul_380, getitem_832);  mul_380 = getitem_832 = None
        triton_kernel_wrapper_functional_proxy_376 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 687, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_512, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_512 = None
        getitem_838 = triton_kernel_wrapper_functional_proxy_376['Y_ptr'];  triton_kernel_wrapper_functional_proxy_376 = None
        view_1836 = torch.ops.aten.view.default(getitem_838, [512, 392, 512]);  getitem_838 = None
        view_1837 = torch.ops.aten.view.default(view_1836, [512, 1024, 14, 14]);  view_1836 = None
        mul_383 = torch.ops.aten.mul.Tensor(add_231, view_1837);  add_231 = view_1837 = None
        empty_542 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_377 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 488, constant_args_idx = 688, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_507, 'S_ptr': getitem_508, 'M_ptr': getitem_509, 'Y_ptr': empty_542, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_507 = getitem_508 = getitem_509 = empty_542 = None
        getitem_839 = triton_kernel_wrapper_functional_proxy_377['Y_ptr'];  triton_kernel_wrapper_functional_proxy_377 = None
        view_1852 = torch.ops.aten.view.default(mul_383, [512, 1024, 196])
        view_1853 = torch.ops.aten.view.default(getitem_839, [512, 392, 512]);  getitem_839 = None
        view_1854 = torch.ops.aten.view.default(view_1853, [512, 1024, 196]);  view_1853 = None
        triton_kernel_wrapper_functional_proxy_378 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 489, constant_args_idx = 689, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1854, 'DY': view_1852, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_840 = triton_kernel_wrapper_functional_proxy_378['DBETA']
        getitem_841 = triton_kernel_wrapper_functional_proxy_378['DGAMMA'];  triton_kernel_wrapper_functional_proxy_378 = None
        empty_543 = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_170 = torch.ops.aten.permute.default(empty_543, [0, 1, 2]);  empty_543 = None
        triton_kernel_wrapper_functional_proxy_379 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 490, constant_args_idx = 690, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1854, 'DY': view_1852, 'INVSTD': rsqrt_39, 'GAMMA': primals_238, 'DBETA': getitem_840, 'DGAMMA': getitem_841, 'DX': permute_170, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1854 = view_1852 = rsqrt_39 = primals_238 = permute_170 = None
        getitem_842 = triton_kernel_wrapper_functional_proxy_379['DX'];  triton_kernel_wrapper_functional_proxy_379 = None
        convert_element_type_default_79 = torch.ops.prims.convert_element_type.default(getitem_841, torch.float32);  getitem_841 = None
        convert_element_type_default_78 = torch.ops.prims.convert_element_type.default(getitem_840, torch.float32);  getitem_840 = None
        empty_544 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_380 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 491, constant_args_idx = 691, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_500, 'S_ptr': getitem_501, 'M_ptr': getitem_502, 'Y_ptr': empty_544, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_500 = getitem_501 = getitem_502 = empty_544 = None
        getitem_843 = triton_kernel_wrapper_functional_proxy_380['Y_ptr'];  triton_kernel_wrapper_functional_proxy_380 = None
        view_1870 = torch.ops.aten.view.default(getitem_842, [512, 1024, 14, 14]);  getitem_842 = None
        empty_545 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_27 = torch.ops.aten.expand.default(empty_545, [512, 256, 14, 14]);  empty_545 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(view_1870, expand_27, convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_27 = convert_element_type_40 = None
        getitem_844 = convolution_backward_26[0];  convolution_backward_26 = None
        empty_546 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_28 = torch.ops.aten.expand.default(empty_546, [1024, 256, 1, 1]);  empty_546 = None
        view_1871 = torch.ops.aten.view.default(getitem_843, [512, 98, 512]);  getitem_843 = None
        view_1872 = torch.ops.aten.view.default(view_1871, [512, 256, 14, 14]);  view_1871 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(view_1870, view_1872, expand_28, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1870 = view_1872 = expand_28 = None
        getitem_848 = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_132 = torch.ops.prims.convert_element_type.default(getitem_848, torch.float32);  getitem_848 = None
        triton_kernel_wrapper_functional_proxy_381 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 692, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_499, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_499 = None
        getitem_850 = triton_kernel_wrapper_functional_proxy_381['Y_ptr'];  triton_kernel_wrapper_functional_proxy_381 = None
        view_1875 = torch.ops.aten.view.default(getitem_850, [512, 98, 512]);  getitem_850 = None
        view_1876 = torch.ops.aten.view.default(view_1875, [512, 256, 14, 14]);  view_1875 = None
        mul_384 = torch.ops.aten.mul.Tensor(getitem_844, view_1876);  getitem_844 = view_1876 = None
        empty_547 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_382 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 492, constant_args_idx = 693, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_494, 'S_ptr': getitem_495, 'M_ptr': getitem_496, 'Y_ptr': empty_547, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_494 = getitem_495 = getitem_496 = empty_547 = None
        getitem_851 = triton_kernel_wrapper_functional_proxy_382['Y_ptr'];  triton_kernel_wrapper_functional_proxy_382 = None
        view_1891 = torch.ops.aten.view.default(mul_384, [512, 256, 196]);  mul_384 = None
        view_1892 = torch.ops.aten.view.default(getitem_851, [512, 98, 512]);  getitem_851 = None
        view_1893 = torch.ops.aten.view.default(view_1892, [512, 256, 196]);  view_1892 = None
        triton_kernel_wrapper_functional_proxy_383 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 493, constant_args_idx = 694, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1893, 'DY': view_1891, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_852 = triton_kernel_wrapper_functional_proxy_383['DBETA']
        getitem_853 = triton_kernel_wrapper_functional_proxy_383['DGAMMA'];  triton_kernel_wrapper_functional_proxy_383 = None
        empty_548 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_171 = torch.ops.aten.permute.default(empty_548, [0, 1, 2]);  empty_548 = None
        triton_kernel_wrapper_functional_proxy_384 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 494, constant_args_idx = 695, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1893, 'DY': view_1891, 'INVSTD': rsqrt_38, 'GAMMA': primals_232, 'DBETA': getitem_852, 'DGAMMA': getitem_853, 'DX': permute_171, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1893 = view_1891 = rsqrt_38 = primals_232 = permute_171 = None
        getitem_854 = triton_kernel_wrapper_functional_proxy_384['DX'];  triton_kernel_wrapper_functional_proxy_384 = None
        convert_element_type_default_77 = torch.ops.prims.convert_element_type.default(getitem_853, torch.float32);  getitem_853 = None
        convert_element_type_default_76 = torch.ops.prims.convert_element_type.default(getitem_852, torch.float32);  getitem_852 = None
        empty_549 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_385 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 495, constant_args_idx = 696, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_487, 'S_ptr': getitem_488, 'M_ptr': getitem_489, 'Y_ptr': empty_549, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_487 = getitem_488 = getitem_489 = empty_549 = None
        getitem_855 = triton_kernel_wrapper_functional_proxy_385['Y_ptr'];  triton_kernel_wrapper_functional_proxy_385 = None
        view_1909 = torch.ops.aten.view.default(getitem_854, [512, 256, 14, 14]);  getitem_854 = None
        empty_550 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_29 = torch.ops.aten.expand.default(empty_550, [512, 256, 14, 14]);  empty_550 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(view_1909, expand_29, convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_29 = convert_element_type_39 = None
        getitem_856 = convolution_backward_28[0];  convolution_backward_28 = None
        empty_551 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_30 = torch.ops.aten.expand.default(empty_551, [256, 256, 3, 3]);  empty_551 = None
        view_1910 = torch.ops.aten.view.default(getitem_855, [512, 98, 512]);  getitem_855 = None
        view_1911 = torch.ops.aten.view.default(view_1910, [512, 256, 14, 14]);  view_1910 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(view_1909, view_1911, expand_30, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_1909 = view_1911 = expand_30 = None
        getitem_860 = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_137 = torch.ops.prims.convert_element_type.default(getitem_860, torch.float32);  getitem_860 = None
        triton_kernel_wrapper_functional_proxy_386 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 697, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_486, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_486 = None
        getitem_862 = triton_kernel_wrapper_functional_proxy_386['Y_ptr'];  triton_kernel_wrapper_functional_proxy_386 = None
        view_1914 = torch.ops.aten.view.default(getitem_862, [512, 98, 512]);  getitem_862 = None
        view_1915 = torch.ops.aten.view.default(view_1914, [512, 256, 14, 14]);  view_1914 = None
        mul_385 = torch.ops.aten.mul.Tensor(getitem_856, view_1915);  getitem_856 = view_1915 = None
        empty_552 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_387 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 496, constant_args_idx = 698, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_481, 'S_ptr': getitem_482, 'M_ptr': getitem_483, 'Y_ptr': empty_552, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_481 = getitem_482 = getitem_483 = empty_552 = None
        getitem_863 = triton_kernel_wrapper_functional_proxy_387['Y_ptr'];  triton_kernel_wrapper_functional_proxy_387 = None
        view_1930 = torch.ops.aten.view.default(mul_385, [512, 256, 196]);  mul_385 = None
        view_1931 = torch.ops.aten.view.default(getitem_863, [512, 98, 512]);  getitem_863 = None
        view_1932 = torch.ops.aten.view.default(view_1931, [512, 256, 196]);  view_1931 = None
        triton_kernel_wrapper_functional_proxy_388 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 497, constant_args_idx = 699, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1932, 'DY': view_1930, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_864 = triton_kernel_wrapper_functional_proxy_388['DBETA']
        getitem_865 = triton_kernel_wrapper_functional_proxy_388['DGAMMA'];  triton_kernel_wrapper_functional_proxy_388 = None
        empty_553 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_172 = torch.ops.aten.permute.default(empty_553, [0, 1, 2]);  empty_553 = None
        triton_kernel_wrapper_functional_proxy_389 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 498, constant_args_idx = 700, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1932, 'DY': view_1930, 'INVSTD': rsqrt_37, 'GAMMA': primals_226, 'DBETA': getitem_864, 'DGAMMA': getitem_865, 'DX': permute_172, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1932 = view_1930 = rsqrt_37 = primals_226 = permute_172 = None
        getitem_866 = triton_kernel_wrapper_functional_proxy_389['DX'];  triton_kernel_wrapper_functional_proxy_389 = None
        convert_element_type_default_75 = torch.ops.prims.convert_element_type.default(getitem_865, torch.float32);  getitem_865 = None
        convert_element_type_default_74 = torch.ops.prims.convert_element_type.default(getitem_864, torch.float32);  getitem_864 = None
        empty_554 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_390 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 499, constant_args_idx = 701, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_474, 'S_ptr': getitem_475, 'M_ptr': getitem_476, 'Y_ptr': empty_554, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_474 = getitem_475 = getitem_476 = empty_554 = None
        getitem_867 = triton_kernel_wrapper_functional_proxy_390['Y_ptr'];  triton_kernel_wrapper_functional_proxy_390 = None
        view_1948 = torch.ops.aten.view.default(getitem_866, [512, 256, 14, 14]);  getitem_866 = None
        empty_555 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_31 = torch.ops.aten.expand.default(empty_555, [512, 1024, 14, 14]);  empty_555 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(view_1948, expand_31, convert_element_type_38, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_31 = convert_element_type_38 = None
        getitem_868 = convolution_backward_30[0];  convolution_backward_30 = None
        empty_556 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_32 = torch.ops.aten.expand.default(empty_556, [256, 1024, 1, 1]);  empty_556 = None
        view_1949 = torch.ops.aten.view.default(getitem_867, [512, 392, 512]);  getitem_867 = None
        view_1950 = torch.ops.aten.view.default(view_1949, [512, 1024, 14, 14]);  view_1949 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(view_1948, view_1950, expand_32, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1948 = view_1950 = expand_32 = None
        getitem_872 = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_142 = torch.ops.prims.convert_element_type.default(getitem_872, torch.float32);  getitem_872 = None
        add_232 = torch.ops.aten.add.Tensor(mul_383, getitem_868);  mul_383 = getitem_868 = None
        triton_kernel_wrapper_functional_proxy_391 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 702, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_473, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_473 = None
        getitem_874 = triton_kernel_wrapper_functional_proxy_391['Y_ptr'];  triton_kernel_wrapper_functional_proxy_391 = None
        view_1953 = torch.ops.aten.view.default(getitem_874, [512, 392, 512]);  getitem_874 = None
        view_1954 = torch.ops.aten.view.default(view_1953, [512, 1024, 14, 14]);  view_1953 = None
        mul_386 = torch.ops.aten.mul.Tensor(add_232, view_1954);  add_232 = view_1954 = None
        empty_557 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_392 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 500, constant_args_idx = 703, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_468, 'S_ptr': getitem_469, 'M_ptr': getitem_470, 'Y_ptr': empty_557, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_468 = getitem_469 = getitem_470 = empty_557 = None
        getitem_875 = triton_kernel_wrapper_functional_proxy_392['Y_ptr'];  triton_kernel_wrapper_functional_proxy_392 = None
        view_1969 = torch.ops.aten.view.default(mul_386, [512, 1024, 196])
        view_1970 = torch.ops.aten.view.default(getitem_875, [512, 392, 512]);  getitem_875 = None
        view_1971 = torch.ops.aten.view.default(view_1970, [512, 1024, 196]);  view_1970 = None
        triton_kernel_wrapper_functional_proxy_393 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 501, constant_args_idx = 704, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1971, 'DY': view_1969, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_876 = triton_kernel_wrapper_functional_proxy_393['DBETA']
        getitem_877 = triton_kernel_wrapper_functional_proxy_393['DGAMMA'];  triton_kernel_wrapper_functional_proxy_393 = None
        empty_558 = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_173 = torch.ops.aten.permute.default(empty_558, [0, 1, 2]);  empty_558 = None
        triton_kernel_wrapper_functional_proxy_394 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 502, constant_args_idx = 705, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_1971, 'DY': view_1969, 'INVSTD': rsqrt_36, 'GAMMA': primals_220, 'DBETA': getitem_876, 'DGAMMA': getitem_877, 'DX': permute_173, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_1971 = view_1969 = rsqrt_36 = primals_220 = permute_173 = None
        getitem_878 = triton_kernel_wrapper_functional_proxy_394['DX'];  triton_kernel_wrapper_functional_proxy_394 = None
        convert_element_type_default_73 = torch.ops.prims.convert_element_type.default(getitem_877, torch.float32);  getitem_877 = None
        convert_element_type_default_72 = torch.ops.prims.convert_element_type.default(getitem_876, torch.float32);  getitem_876 = None
        empty_559 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_395 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 503, constant_args_idx = 706, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_461, 'S_ptr': getitem_462, 'M_ptr': getitem_463, 'Y_ptr': empty_559, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_461 = getitem_462 = getitem_463 = empty_559 = None
        getitem_879 = triton_kernel_wrapper_functional_proxy_395['Y_ptr'];  triton_kernel_wrapper_functional_proxy_395 = None
        view_1987 = torch.ops.aten.view.default(getitem_878, [512, 1024, 14, 14]);  getitem_878 = None
        empty_560 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_33 = torch.ops.aten.expand.default(empty_560, [512, 256, 14, 14]);  empty_560 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_1987, expand_33, convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_33 = convert_element_type_37 = None
        getitem_880 = convolution_backward_32[0];  convolution_backward_32 = None
        empty_561 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_34 = torch.ops.aten.expand.default(empty_561, [1024, 256, 1, 1]);  empty_561 = None
        view_1988 = torch.ops.aten.view.default(getitem_879, [512, 98, 512]);  getitem_879 = None
        view_1989 = torch.ops.aten.view.default(view_1988, [512, 256, 14, 14]);  view_1988 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(view_1987, view_1989, expand_34, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_1987 = view_1989 = expand_34 = None
        getitem_884 = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_147 = torch.ops.prims.convert_element_type.default(getitem_884, torch.float32);  getitem_884 = None
        triton_kernel_wrapper_functional_proxy_396 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 707, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_460, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_460 = None
        getitem_886 = triton_kernel_wrapper_functional_proxy_396['Y_ptr'];  triton_kernel_wrapper_functional_proxy_396 = None
        view_1992 = torch.ops.aten.view.default(getitem_886, [512, 98, 512]);  getitem_886 = None
        view_1993 = torch.ops.aten.view.default(view_1992, [512, 256, 14, 14]);  view_1992 = None
        mul_387 = torch.ops.aten.mul.Tensor(getitem_880, view_1993);  getitem_880 = view_1993 = None
        empty_562 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_397 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 504, constant_args_idx = 708, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_455, 'S_ptr': getitem_456, 'M_ptr': getitem_457, 'Y_ptr': empty_562, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_455 = getitem_456 = getitem_457 = empty_562 = None
        getitem_887 = triton_kernel_wrapper_functional_proxy_397['Y_ptr'];  triton_kernel_wrapper_functional_proxy_397 = None
        view_2008 = torch.ops.aten.view.default(mul_387, [512, 256, 196]);  mul_387 = None
        view_2009 = torch.ops.aten.view.default(getitem_887, [512, 98, 512]);  getitem_887 = None
        view_2010 = torch.ops.aten.view.default(view_2009, [512, 256, 196]);  view_2009 = None
        triton_kernel_wrapper_functional_proxy_398 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 505, constant_args_idx = 709, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2010, 'DY': view_2008, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_888 = triton_kernel_wrapper_functional_proxy_398['DBETA']
        getitem_889 = triton_kernel_wrapper_functional_proxy_398['DGAMMA'];  triton_kernel_wrapper_functional_proxy_398 = None
        empty_563 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_174 = torch.ops.aten.permute.default(empty_563, [0, 1, 2]);  empty_563 = None
        triton_kernel_wrapper_functional_proxy_399 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 506, constant_args_idx = 710, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2010, 'DY': view_2008, 'INVSTD': rsqrt_35, 'GAMMA': primals_214, 'DBETA': getitem_888, 'DGAMMA': getitem_889, 'DX': permute_174, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2010 = view_2008 = rsqrt_35 = primals_214 = permute_174 = None
        getitem_890 = triton_kernel_wrapper_functional_proxy_399['DX'];  triton_kernel_wrapper_functional_proxy_399 = None
        convert_element_type_default_71 = torch.ops.prims.convert_element_type.default(getitem_889, torch.float32);  getitem_889 = None
        convert_element_type_default_70 = torch.ops.prims.convert_element_type.default(getitem_888, torch.float32);  getitem_888 = None
        empty_564 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_400 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 507, constant_args_idx = 711, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_448, 'S_ptr': getitem_449, 'M_ptr': getitem_450, 'Y_ptr': empty_564, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_448 = getitem_449 = getitem_450 = empty_564 = None
        getitem_891 = triton_kernel_wrapper_functional_proxy_400['Y_ptr'];  triton_kernel_wrapper_functional_proxy_400 = None
        view_2026 = torch.ops.aten.view.default(getitem_890, [512, 256, 14, 14]);  getitem_890 = None
        empty_565 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_35 = torch.ops.aten.expand.default(empty_565, [512, 256, 14, 14]);  empty_565 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(view_2026, expand_35, convert_element_type_36, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_35 = convert_element_type_36 = None
        getitem_892 = convolution_backward_34[0];  convolution_backward_34 = None
        empty_566 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_36 = torch.ops.aten.expand.default(empty_566, [256, 256, 3, 3]);  empty_566 = None
        view_2027 = torch.ops.aten.view.default(getitem_891, [512, 98, 512]);  getitem_891 = None
        view_2028 = torch.ops.aten.view.default(view_2027, [512, 256, 14, 14]);  view_2027 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(view_2026, view_2028, expand_36, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2026 = view_2028 = expand_36 = None
        getitem_896 = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_152 = torch.ops.prims.convert_element_type.default(getitem_896, torch.float32);  getitem_896 = None
        triton_kernel_wrapper_functional_proxy_401 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 712, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_447, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_447 = None
        getitem_898 = triton_kernel_wrapper_functional_proxy_401['Y_ptr'];  triton_kernel_wrapper_functional_proxy_401 = None
        view_2031 = torch.ops.aten.view.default(getitem_898, [512, 98, 512]);  getitem_898 = None
        view_2032 = torch.ops.aten.view.default(view_2031, [512, 256, 14, 14]);  view_2031 = None
        mul_388 = torch.ops.aten.mul.Tensor(getitem_892, view_2032);  getitem_892 = view_2032 = None
        empty_567 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_402 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 508, constant_args_idx = 713, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_442, 'S_ptr': getitem_443, 'M_ptr': getitem_444, 'Y_ptr': empty_567, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_442 = getitem_443 = getitem_444 = empty_567 = None
        getitem_899 = triton_kernel_wrapper_functional_proxy_402['Y_ptr'];  triton_kernel_wrapper_functional_proxy_402 = None
        view_2047 = torch.ops.aten.view.default(mul_388, [512, 256, 196]);  mul_388 = None
        view_2048 = torch.ops.aten.view.default(getitem_899, [512, 98, 512]);  getitem_899 = None
        view_2049 = torch.ops.aten.view.default(view_2048, [512, 256, 196]);  view_2048 = None
        triton_kernel_wrapper_functional_proxy_403 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 509, constant_args_idx = 714, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2049, 'DY': view_2047, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_900 = triton_kernel_wrapper_functional_proxy_403['DBETA']
        getitem_901 = triton_kernel_wrapper_functional_proxy_403['DGAMMA'];  triton_kernel_wrapper_functional_proxy_403 = None
        empty_568 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_175 = torch.ops.aten.permute.default(empty_568, [0, 1, 2]);  empty_568 = None
        triton_kernel_wrapper_functional_proxy_404 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 510, constant_args_idx = 715, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2049, 'DY': view_2047, 'INVSTD': rsqrt_34, 'GAMMA': primals_208, 'DBETA': getitem_900, 'DGAMMA': getitem_901, 'DX': permute_175, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2049 = view_2047 = rsqrt_34 = primals_208 = permute_175 = None
        getitem_902 = triton_kernel_wrapper_functional_proxy_404['DX'];  triton_kernel_wrapper_functional_proxy_404 = None
        convert_element_type_default_69 = torch.ops.prims.convert_element_type.default(getitem_901, torch.float32);  getitem_901 = None
        convert_element_type_default_68 = torch.ops.prims.convert_element_type.default(getitem_900, torch.float32);  getitem_900 = None
        empty_569 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_405 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 511, constant_args_idx = 716, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_435, 'S_ptr': getitem_436, 'M_ptr': getitem_437, 'Y_ptr': empty_569, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_435 = getitem_436 = getitem_437 = empty_569 = None
        getitem_903 = triton_kernel_wrapper_functional_proxy_405['Y_ptr'];  triton_kernel_wrapper_functional_proxy_405 = None
        view_2065 = torch.ops.aten.view.default(getitem_902, [512, 256, 14, 14]);  getitem_902 = None
        empty_570 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_37 = torch.ops.aten.expand.default(empty_570, [512, 1024, 14, 14]);  empty_570 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(view_2065, expand_37, convert_element_type_35, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_37 = convert_element_type_35 = None
        getitem_904 = convolution_backward_36[0];  convolution_backward_36 = None
        empty_571 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_38 = torch.ops.aten.expand.default(empty_571, [256, 1024, 1, 1]);  empty_571 = None
        view_2066 = torch.ops.aten.view.default(getitem_903, [512, 392, 512]);  getitem_903 = None
        view_2067 = torch.ops.aten.view.default(view_2066, [512, 1024, 14, 14]);  view_2066 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(view_2065, view_2067, expand_38, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2065 = view_2067 = expand_38 = None
        getitem_908 = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_157 = torch.ops.prims.convert_element_type.default(getitem_908, torch.float32);  getitem_908 = None
        add_233 = torch.ops.aten.add.Tensor(mul_386, getitem_904);  mul_386 = getitem_904 = None
        triton_kernel_wrapper_functional_proxy_406 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 717, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_434, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_434 = None
        getitem_910 = triton_kernel_wrapper_functional_proxy_406['Y_ptr'];  triton_kernel_wrapper_functional_proxy_406 = None
        view_2070 = torch.ops.aten.view.default(getitem_910, [512, 392, 512]);  getitem_910 = None
        view_2071 = torch.ops.aten.view.default(view_2070, [512, 1024, 14, 14]);  view_2070 = None
        mul_389 = torch.ops.aten.mul.Tensor(add_233, view_2071);  add_233 = view_2071 = None
        empty_572 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_407 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 512, constant_args_idx = 718, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_429, 'S_ptr': getitem_430, 'M_ptr': getitem_431, 'Y_ptr': empty_572, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_429 = getitem_430 = getitem_431 = empty_572 = None
        getitem_911 = triton_kernel_wrapper_functional_proxy_407['Y_ptr'];  triton_kernel_wrapper_functional_proxy_407 = None
        view_2086 = torch.ops.aten.view.default(mul_389, [512, 1024, 196])
        view_2087 = torch.ops.aten.view.default(getitem_911, [512, 392, 512]);  getitem_911 = None
        view_2088 = torch.ops.aten.view.default(view_2087, [512, 1024, 196]);  view_2087 = None
        triton_kernel_wrapper_functional_proxy_408 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 513, constant_args_idx = 719, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2088, 'DY': view_2086, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_912 = triton_kernel_wrapper_functional_proxy_408['DBETA']
        getitem_913 = triton_kernel_wrapper_functional_proxy_408['DGAMMA'];  triton_kernel_wrapper_functional_proxy_408 = None
        empty_573 = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_176 = torch.ops.aten.permute.default(empty_573, [0, 1, 2]);  empty_573 = None
        triton_kernel_wrapper_functional_proxy_409 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 514, constant_args_idx = 720, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2088, 'DY': view_2086, 'INVSTD': rsqrt_33, 'GAMMA': primals_202, 'DBETA': getitem_912, 'DGAMMA': getitem_913, 'DX': permute_176, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2088 = view_2086 = rsqrt_33 = primals_202 = permute_176 = None
        getitem_914 = triton_kernel_wrapper_functional_proxy_409['DX'];  triton_kernel_wrapper_functional_proxy_409 = None
        convert_element_type_default_67 = torch.ops.prims.convert_element_type.default(getitem_913, torch.float32);  getitem_913 = None
        convert_element_type_default_66 = torch.ops.prims.convert_element_type.default(getitem_912, torch.float32);  getitem_912 = None
        empty_574 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_410 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 515, constant_args_idx = 721, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_422, 'S_ptr': getitem_423, 'M_ptr': getitem_424, 'Y_ptr': empty_574, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_422 = getitem_423 = getitem_424 = empty_574 = None
        getitem_915 = triton_kernel_wrapper_functional_proxy_410['Y_ptr'];  triton_kernel_wrapper_functional_proxy_410 = None
        view_2104 = torch.ops.aten.view.default(getitem_914, [512, 1024, 14, 14]);  getitem_914 = None
        empty_575 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_39 = torch.ops.aten.expand.default(empty_575, [512, 256, 14, 14]);  empty_575 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(view_2104, expand_39, convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_39 = convert_element_type_34 = None
        getitem_916 = convolution_backward_38[0];  convolution_backward_38 = None
        empty_576 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_40 = torch.ops.aten.expand.default(empty_576, [1024, 256, 1, 1]);  empty_576 = None
        view_2105 = torch.ops.aten.view.default(getitem_915, [512, 98, 512]);  getitem_915 = None
        view_2106 = torch.ops.aten.view.default(view_2105, [512, 256, 14, 14]);  view_2105 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(view_2104, view_2106, expand_40, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2104 = view_2106 = expand_40 = None
        getitem_920 = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_162 = torch.ops.prims.convert_element_type.default(getitem_920, torch.float32);  getitem_920 = None
        triton_kernel_wrapper_functional_proxy_411 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 722, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_421, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_421 = None
        getitem_922 = triton_kernel_wrapper_functional_proxy_411['Y_ptr'];  triton_kernel_wrapper_functional_proxy_411 = None
        view_2109 = torch.ops.aten.view.default(getitem_922, [512, 98, 512]);  getitem_922 = None
        view_2110 = torch.ops.aten.view.default(view_2109, [512, 256, 14, 14]);  view_2109 = None
        mul_390 = torch.ops.aten.mul.Tensor(getitem_916, view_2110);  getitem_916 = view_2110 = None
        empty_577 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_412 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 516, constant_args_idx = 723, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_416, 'S_ptr': getitem_417, 'M_ptr': getitem_418, 'Y_ptr': empty_577, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_416 = getitem_417 = getitem_418 = empty_577 = None
        getitem_923 = triton_kernel_wrapper_functional_proxy_412['Y_ptr'];  triton_kernel_wrapper_functional_proxy_412 = None
        view_2125 = torch.ops.aten.view.default(mul_390, [512, 256, 196]);  mul_390 = None
        view_2126 = torch.ops.aten.view.default(getitem_923, [512, 98, 512]);  getitem_923 = None
        view_2127 = torch.ops.aten.view.default(view_2126, [512, 256, 196]);  view_2126 = None
        triton_kernel_wrapper_functional_proxy_413 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 517, constant_args_idx = 724, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2127, 'DY': view_2125, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_924 = triton_kernel_wrapper_functional_proxy_413['DBETA']
        getitem_925 = triton_kernel_wrapper_functional_proxy_413['DGAMMA'];  triton_kernel_wrapper_functional_proxy_413 = None
        empty_578 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_177 = torch.ops.aten.permute.default(empty_578, [0, 1, 2]);  empty_578 = None
        triton_kernel_wrapper_functional_proxy_414 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 518, constant_args_idx = 725, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2127, 'DY': view_2125, 'INVSTD': rsqrt_32, 'GAMMA': primals_196, 'DBETA': getitem_924, 'DGAMMA': getitem_925, 'DX': permute_177, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2127 = view_2125 = rsqrt_32 = primals_196 = permute_177 = None
        getitem_926 = triton_kernel_wrapper_functional_proxy_414['DX'];  triton_kernel_wrapper_functional_proxy_414 = None
        convert_element_type_default_65 = torch.ops.prims.convert_element_type.default(getitem_925, torch.float32);  getitem_925 = None
        convert_element_type_default_64 = torch.ops.prims.convert_element_type.default(getitem_924, torch.float32);  getitem_924 = None
        empty_579 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_415 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 519, constant_args_idx = 726, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_409, 'S_ptr': getitem_410, 'M_ptr': getitem_411, 'Y_ptr': empty_579, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_409 = getitem_410 = getitem_411 = empty_579 = None
        getitem_927 = triton_kernel_wrapper_functional_proxy_415['Y_ptr'];  triton_kernel_wrapper_functional_proxy_415 = None
        view_2143 = torch.ops.aten.view.default(getitem_926, [512, 256, 14, 14]);  getitem_926 = None
        empty_580 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_41 = torch.ops.aten.expand.default(empty_580, [512, 256, 14, 14]);  empty_580 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(view_2143, expand_41, convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_41 = convert_element_type_33 = None
        getitem_928 = convolution_backward_40[0];  convolution_backward_40 = None
        empty_581 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_42 = torch.ops.aten.expand.default(empty_581, [256, 256, 3, 3]);  empty_581 = None
        view_2144 = torch.ops.aten.view.default(getitem_927, [512, 98, 512]);  getitem_927 = None
        view_2145 = torch.ops.aten.view.default(view_2144, [512, 256, 14, 14]);  view_2144 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(view_2143, view_2145, expand_42, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2143 = view_2145 = expand_42 = None
        getitem_932 = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(getitem_932, torch.float32);  getitem_932 = None
        triton_kernel_wrapper_functional_proxy_416 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 727, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_408, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_408 = None
        getitem_934 = triton_kernel_wrapper_functional_proxy_416['Y_ptr'];  triton_kernel_wrapper_functional_proxy_416 = None
        view_2148 = torch.ops.aten.view.default(getitem_934, [512, 98, 512]);  getitem_934 = None
        view_2149 = torch.ops.aten.view.default(view_2148, [512, 256, 14, 14]);  view_2148 = None
        mul_391 = torch.ops.aten.mul.Tensor(getitem_928, view_2149);  getitem_928 = view_2149 = None
        empty_582 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_417 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 520, constant_args_idx = 728, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_403, 'S_ptr': getitem_404, 'M_ptr': getitem_405, 'Y_ptr': empty_582, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_403 = getitem_404 = getitem_405 = empty_582 = None
        getitem_935 = triton_kernel_wrapper_functional_proxy_417['Y_ptr'];  triton_kernel_wrapper_functional_proxy_417 = None
        view_2164 = torch.ops.aten.view.default(mul_391, [512, 256, 196]);  mul_391 = None
        view_2165 = torch.ops.aten.view.default(getitem_935, [512, 98, 512]);  getitem_935 = None
        view_2166 = torch.ops.aten.view.default(view_2165, [512, 256, 196]);  view_2165 = None
        triton_kernel_wrapper_functional_proxy_418 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 521, constant_args_idx = 729, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2166, 'DY': view_2164, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_936 = triton_kernel_wrapper_functional_proxy_418['DBETA']
        getitem_937 = triton_kernel_wrapper_functional_proxy_418['DGAMMA'];  triton_kernel_wrapper_functional_proxy_418 = None
        empty_583 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_178 = torch.ops.aten.permute.default(empty_583, [0, 1, 2]);  empty_583 = None
        triton_kernel_wrapper_functional_proxy_419 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 522, constant_args_idx = 730, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2166, 'DY': view_2164, 'INVSTD': rsqrt_31, 'GAMMA': primals_190, 'DBETA': getitem_936, 'DGAMMA': getitem_937, 'DX': permute_178, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2166 = view_2164 = rsqrt_31 = primals_190 = permute_178 = None
        getitem_938 = triton_kernel_wrapper_functional_proxy_419['DX'];  triton_kernel_wrapper_functional_proxy_419 = None
        convert_element_type_default_63 = torch.ops.prims.convert_element_type.default(getitem_937, torch.float32);  getitem_937 = None
        convert_element_type_default_62 = torch.ops.prims.convert_element_type.default(getitem_936, torch.float32);  getitem_936 = None
        empty_584 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_420 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 523, constant_args_idx = 731, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_396, 'S_ptr': getitem_397, 'M_ptr': getitem_398, 'Y_ptr': empty_584, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_396 = getitem_397 = getitem_398 = empty_584 = None
        getitem_939 = triton_kernel_wrapper_functional_proxy_420['Y_ptr'];  triton_kernel_wrapper_functional_proxy_420 = None
        view_2182 = torch.ops.aten.view.default(getitem_938, [512, 256, 14, 14]);  getitem_938 = None
        empty_585 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_43 = torch.ops.aten.expand.default(empty_585, [512, 1024, 14, 14]);  empty_585 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(view_2182, expand_43, convert_element_type_32, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_43 = convert_element_type_32 = None
        getitem_940 = convolution_backward_42[0];  convolution_backward_42 = None
        empty_586 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_44 = torch.ops.aten.expand.default(empty_586, [256, 1024, 1, 1]);  empty_586 = None
        view_2183 = torch.ops.aten.view.default(getitem_939, [512, 392, 512]);  getitem_939 = None
        view_2184 = torch.ops.aten.view.default(view_2183, [512, 1024, 14, 14]);  view_2183 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(view_2182, view_2184, expand_44, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2182 = view_2184 = expand_44 = None
        getitem_944 = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(getitem_944, torch.float32);  getitem_944 = None
        add_234 = torch.ops.aten.add.Tensor(mul_389, getitem_940);  mul_389 = getitem_940 = None
        triton_kernel_wrapper_functional_proxy_421 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 732, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_395, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_395 = None
        getitem_946 = triton_kernel_wrapper_functional_proxy_421['Y_ptr'];  triton_kernel_wrapper_functional_proxy_421 = None
        view_2187 = torch.ops.aten.view.default(getitem_946, [512, 392, 512]);  getitem_946 = None
        view_2188 = torch.ops.aten.view.default(view_2187, [512, 1024, 14, 14]);  view_2187 = None
        mul_392 = torch.ops.aten.mul.Tensor(add_234, view_2188);  add_234 = view_2188 = None
        empty_587 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_422 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 524, constant_args_idx = 733, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_390, 'S_ptr': getitem_391, 'M_ptr': getitem_392, 'Y_ptr': empty_587, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_390 = getitem_391 = getitem_392 = empty_587 = None
        getitem_947 = triton_kernel_wrapper_functional_proxy_422['Y_ptr'];  triton_kernel_wrapper_functional_proxy_422 = None
        view_2203 = torch.ops.aten.view.default(mul_392, [512, 1024, 196])
        view_2204 = torch.ops.aten.view.default(getitem_947, [512, 392, 512]);  getitem_947 = None
        view_2205 = torch.ops.aten.view.default(view_2204, [512, 1024, 196]);  view_2204 = None
        triton_kernel_wrapper_functional_proxy_423 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 525, constant_args_idx = 734, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2205, 'DY': view_2203, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_948 = triton_kernel_wrapper_functional_proxy_423['DBETA']
        getitem_949 = triton_kernel_wrapper_functional_proxy_423['DGAMMA'];  triton_kernel_wrapper_functional_proxy_423 = None
        empty_588 = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_179 = torch.ops.aten.permute.default(empty_588, [0, 1, 2]);  empty_588 = None
        triton_kernel_wrapper_functional_proxy_424 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 526, constant_args_idx = 735, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2205, 'DY': view_2203, 'INVSTD': rsqrt_30, 'GAMMA': primals_184, 'DBETA': getitem_948, 'DGAMMA': getitem_949, 'DX': permute_179, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2205 = view_2203 = rsqrt_30 = primals_184 = permute_179 = None
        getitem_950 = triton_kernel_wrapper_functional_proxy_424['DX'];  triton_kernel_wrapper_functional_proxy_424 = None
        convert_element_type_default_61 = torch.ops.prims.convert_element_type.default(getitem_949, torch.float32);  getitem_949 = None
        convert_element_type_default_60 = torch.ops.prims.convert_element_type.default(getitem_948, torch.float32);  getitem_948 = None
        empty_589 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_425 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 527, constant_args_idx = 736, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_383, 'S_ptr': getitem_384, 'M_ptr': getitem_385, 'Y_ptr': empty_589, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_383 = getitem_384 = getitem_385 = empty_589 = None
        getitem_951 = triton_kernel_wrapper_functional_proxy_425['Y_ptr'];  triton_kernel_wrapper_functional_proxy_425 = None
        view_2221 = torch.ops.aten.view.default(getitem_950, [512, 1024, 14, 14]);  getitem_950 = None
        empty_590 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_45 = torch.ops.aten.expand.default(empty_590, [512, 256, 14, 14]);  empty_590 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(view_2221, expand_45, convert_element_type_31, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_45 = convert_element_type_31 = None
        getitem_952 = convolution_backward_44[0];  convolution_backward_44 = None
        empty_591 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_46 = torch.ops.aten.expand.default(empty_591, [1024, 256, 1, 1]);  empty_591 = None
        view_2222 = torch.ops.aten.view.default(getitem_951, [512, 98, 512]);  getitem_951 = None
        view_2223 = torch.ops.aten.view.default(view_2222, [512, 256, 14, 14]);  view_2222 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(view_2221, view_2223, expand_46, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2221 = view_2223 = expand_46 = None
        getitem_956 = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_177 = torch.ops.prims.convert_element_type.default(getitem_956, torch.float32);  getitem_956 = None
        triton_kernel_wrapper_functional_proxy_426 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 737, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_382, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_382 = None
        getitem_958 = triton_kernel_wrapper_functional_proxy_426['Y_ptr'];  triton_kernel_wrapper_functional_proxy_426 = None
        view_2226 = torch.ops.aten.view.default(getitem_958, [512, 98, 512]);  getitem_958 = None
        view_2227 = torch.ops.aten.view.default(view_2226, [512, 256, 14, 14]);  view_2226 = None
        mul_393 = torch.ops.aten.mul.Tensor(getitem_952, view_2227);  getitem_952 = view_2227 = None
        empty_592 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_427 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 528, constant_args_idx = 738, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_377, 'S_ptr': getitem_378, 'M_ptr': getitem_379, 'Y_ptr': empty_592, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_377 = getitem_378 = getitem_379 = empty_592 = None
        getitem_959 = triton_kernel_wrapper_functional_proxy_427['Y_ptr'];  triton_kernel_wrapper_functional_proxy_427 = None
        view_2242 = torch.ops.aten.view.default(mul_393, [512, 256, 196]);  mul_393 = None
        view_2243 = torch.ops.aten.view.default(getitem_959, [512, 98, 512]);  getitem_959 = None
        view_2244 = torch.ops.aten.view.default(view_2243, [512, 256, 196]);  view_2243 = None
        triton_kernel_wrapper_functional_proxy_428 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 529, constant_args_idx = 739, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2244, 'DY': view_2242, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_960 = triton_kernel_wrapper_functional_proxy_428['DBETA']
        getitem_961 = triton_kernel_wrapper_functional_proxy_428['DGAMMA'];  triton_kernel_wrapper_functional_proxy_428 = None
        empty_593 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_180 = torch.ops.aten.permute.default(empty_593, [0, 1, 2]);  empty_593 = None
        triton_kernel_wrapper_functional_proxy_429 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 530, constant_args_idx = 740, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2244, 'DY': view_2242, 'INVSTD': rsqrt_29, 'GAMMA': primals_178, 'DBETA': getitem_960, 'DGAMMA': getitem_961, 'DX': permute_180, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2244 = view_2242 = rsqrt_29 = primals_178 = permute_180 = None
        getitem_962 = triton_kernel_wrapper_functional_proxy_429['DX'];  triton_kernel_wrapper_functional_proxy_429 = None
        convert_element_type_default_59 = torch.ops.prims.convert_element_type.default(getitem_961, torch.float32);  getitem_961 = None
        convert_element_type_default_58 = torch.ops.prims.convert_element_type.default(getitem_960, torch.float32);  getitem_960 = None
        empty_594 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_430 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 531, constant_args_idx = 741, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_370, 'S_ptr': getitem_371, 'M_ptr': getitem_372, 'Y_ptr': empty_594, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_370 = getitem_371 = getitem_372 = empty_594 = None
        getitem_963 = triton_kernel_wrapper_functional_proxy_430['Y_ptr'];  triton_kernel_wrapper_functional_proxy_430 = None
        view_2260 = torch.ops.aten.view.default(getitem_962, [512, 256, 14, 14]);  getitem_962 = None
        empty_595 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_47 = torch.ops.aten.expand.default(empty_595, [512, 256, 14, 14]);  empty_595 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(view_2260, expand_47, convert_element_type_30, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_47 = convert_element_type_30 = None
        getitem_964 = convolution_backward_46[0];  convolution_backward_46 = None
        empty_596 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_48 = torch.ops.aten.expand.default(empty_596, [256, 256, 3, 3]);  empty_596 = None
        view_2261 = torch.ops.aten.view.default(getitem_963, [512, 98, 512]);  getitem_963 = None
        view_2262 = torch.ops.aten.view.default(view_2261, [512, 256, 14, 14]);  view_2261 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(view_2260, view_2262, expand_48, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2260 = view_2262 = expand_48 = None
        getitem_968 = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(getitem_968, torch.float32);  getitem_968 = None
        triton_kernel_wrapper_functional_proxy_431 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 742, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_369, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_369 = None
        getitem_970 = triton_kernel_wrapper_functional_proxy_431['Y_ptr'];  triton_kernel_wrapper_functional_proxy_431 = None
        view_2265 = torch.ops.aten.view.default(getitem_970, [512, 98, 512]);  getitem_970 = None
        view_2266 = torch.ops.aten.view.default(view_2265, [512, 256, 14, 14]);  view_2265 = None
        mul_394 = torch.ops.aten.mul.Tensor(getitem_964, view_2266);  getitem_964 = view_2266 = None
        empty_597 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_432 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 532, constant_args_idx = 743, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_364, 'S_ptr': getitem_365, 'M_ptr': getitem_366, 'Y_ptr': empty_597, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_364 = getitem_365 = getitem_366 = empty_597 = None
        getitem_971 = triton_kernel_wrapper_functional_proxy_432['Y_ptr'];  triton_kernel_wrapper_functional_proxy_432 = None
        view_2281 = torch.ops.aten.view.default(mul_394, [512, 256, 196]);  mul_394 = None
        view_2282 = torch.ops.aten.view.default(getitem_971, [512, 98, 512]);  getitem_971 = None
        view_2283 = torch.ops.aten.view.default(view_2282, [512, 256, 196]);  view_2282 = None
        triton_kernel_wrapper_functional_proxy_433 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 533, constant_args_idx = 744, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2283, 'DY': view_2281, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_972 = triton_kernel_wrapper_functional_proxy_433['DBETA']
        getitem_973 = triton_kernel_wrapper_functional_proxy_433['DGAMMA'];  triton_kernel_wrapper_functional_proxy_433 = None
        empty_598 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_181 = torch.ops.aten.permute.default(empty_598, [0, 1, 2]);  empty_598 = None
        triton_kernel_wrapper_functional_proxy_434 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 534, constant_args_idx = 745, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2283, 'DY': view_2281, 'INVSTD': rsqrt_28, 'GAMMA': primals_172, 'DBETA': getitem_972, 'DGAMMA': getitem_973, 'DX': permute_181, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2283 = view_2281 = rsqrt_28 = primals_172 = permute_181 = None
        getitem_974 = triton_kernel_wrapper_functional_proxy_434['DX'];  triton_kernel_wrapper_functional_proxy_434 = None
        convert_element_type_default_57 = torch.ops.prims.convert_element_type.default(getitem_973, torch.float32);  getitem_973 = None
        convert_element_type_default_56 = torch.ops.prims.convert_element_type.default(getitem_972, torch.float32);  getitem_972 = None
        empty_599 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_435 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 535, constant_args_idx = 746, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_357, 'S_ptr': getitem_358, 'M_ptr': getitem_359, 'Y_ptr': empty_599, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_357 = getitem_358 = getitem_359 = empty_599 = None
        getitem_975 = triton_kernel_wrapper_functional_proxy_435['Y_ptr'];  triton_kernel_wrapper_functional_proxy_435 = None
        view_2299 = torch.ops.aten.view.default(getitem_974, [512, 256, 14, 14]);  getitem_974 = None
        empty_600 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_49 = torch.ops.aten.expand.default(empty_600, [512, 1024, 14, 14]);  empty_600 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(view_2299, expand_49, convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_49 = convert_element_type_29 = None
        getitem_976 = convolution_backward_48[0];  convolution_backward_48 = None
        empty_601 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_50 = torch.ops.aten.expand.default(empty_601, [256, 1024, 1, 1]);  empty_601 = None
        view_2300 = torch.ops.aten.view.default(getitem_975, [512, 392, 512]);  getitem_975 = None
        view_2301 = torch.ops.aten.view.default(view_2300, [512, 1024, 14, 14]);  view_2300 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(view_2299, view_2301, expand_50, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2299 = view_2301 = expand_50 = None
        getitem_980 = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(getitem_980, torch.float32);  getitem_980 = None
        add_235 = torch.ops.aten.add.Tensor(mul_392, getitem_976);  mul_392 = getitem_976 = None
        triton_kernel_wrapper_functional_proxy_436 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 747, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_356, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_356 = None
        getitem_982 = triton_kernel_wrapper_functional_proxy_436['Y_ptr'];  triton_kernel_wrapper_functional_proxy_436 = None
        view_2304 = torch.ops.aten.view.default(getitem_982, [512, 392, 512]);  getitem_982 = None
        view_2305 = torch.ops.aten.view.default(view_2304, [512, 1024, 14, 14]);  view_2304 = None
        mul_395 = torch.ops.aten.mul.Tensor(add_235, view_2305);  add_235 = view_2305 = None
        empty_602 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_437 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 536, constant_args_idx = 748, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_351, 'S_ptr': getitem_352, 'M_ptr': getitem_353, 'Y_ptr': empty_602, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_351 = getitem_352 = getitem_353 = empty_602 = None
        getitem_983 = triton_kernel_wrapper_functional_proxy_437['Y_ptr'];  triton_kernel_wrapper_functional_proxy_437 = None
        view_2320 = torch.ops.aten.view.default(mul_395, [512, 1024, 196]);  mul_395 = None
        view_2321 = torch.ops.aten.view.default(getitem_983, [512, 392, 512]);  getitem_983 = None
        view_2322 = torch.ops.aten.view.default(view_2321, [512, 1024, 196]);  view_2321 = None
        triton_kernel_wrapper_functional_proxy_438 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 537, constant_args_idx = 749, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2322, 'DY': view_2320, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_984 = triton_kernel_wrapper_functional_proxy_438['DBETA']
        getitem_985 = triton_kernel_wrapper_functional_proxy_438['DGAMMA'];  triton_kernel_wrapper_functional_proxy_438 = None
        empty_603 = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_182 = torch.ops.aten.permute.default(empty_603, [0, 1, 2]);  empty_603 = None
        triton_kernel_wrapper_functional_proxy_439 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 538, constant_args_idx = 750, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2322, 'DY': view_2320, 'INVSTD': rsqrt_27, 'GAMMA': primals_166, 'DBETA': getitem_984, 'DGAMMA': getitem_985, 'DX': permute_182, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2322 = rsqrt_27 = primals_166 = permute_182 = None
        getitem_986 = triton_kernel_wrapper_functional_proxy_439['DX'];  triton_kernel_wrapper_functional_proxy_439 = None
        convert_element_type_default_55 = torch.ops.prims.convert_element_type.default(getitem_985, torch.float32);  getitem_985 = None
        convert_element_type_default_54 = torch.ops.prims.convert_element_type.default(getitem_984, torch.float32);  getitem_984 = None
        empty_604 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_440 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 539, constant_args_idx = 751, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_344, 'S_ptr': getitem_345, 'M_ptr': getitem_346, 'Y_ptr': empty_604, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_344 = getitem_345 = getitem_346 = empty_604 = None
        getitem_987 = triton_kernel_wrapper_functional_proxy_440['Y_ptr'];  triton_kernel_wrapper_functional_proxy_440 = None
        view_2338 = torch.ops.aten.view.default(getitem_986, [512, 1024, 14, 14]);  getitem_986 = None
        empty_605 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_51 = torch.ops.aten.expand.default(empty_605, [512, 512, 28, 28]);  empty_605 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(view_2338, expand_51, convert_element_type_28, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_51 = convert_element_type_28 = None
        getitem_988 = convolution_backward_50[0];  convolution_backward_50 = None
        empty_606 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_52 = torch.ops.aten.expand.default(empty_606, [1024, 512, 1, 1]);  empty_606 = None
        view_2339 = torch.ops.aten.view.default(getitem_987, [512, 784, 512]);  getitem_987 = None
        view_2340 = torch.ops.aten.view.default(view_2339, [512, 512, 28, 28]);  view_2339 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(view_2338, view_2340, expand_52, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2338 = view_2340 = expand_52 = None
        getitem_992 = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(getitem_992, torch.float32);  getitem_992 = None
        empty_607 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_441 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 540, constant_args_idx = 752, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_341, 'S_ptr': getitem_342, 'M_ptr': getitem_343, 'Y_ptr': empty_607, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_341 = getitem_342 = getitem_343 = empty_607 = None
        getitem_994 = triton_kernel_wrapper_functional_proxy_441['Y_ptr'];  triton_kernel_wrapper_functional_proxy_441 = None
        view_2356 = torch.ops.aten.view.default(getitem_994, [512, 392, 512]);  getitem_994 = None
        view_2357 = torch.ops.aten.view.default(view_2356, [512, 1024, 196]);  view_2356 = None
        triton_kernel_wrapper_functional_proxy_442 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 541, constant_args_idx = 753, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2357, 'DY': view_2320, 'DBETA': full_default_152, 'DGAMMA': full_default_152, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_152 = None
        getitem_995 = triton_kernel_wrapper_functional_proxy_442['DBETA']
        getitem_996 = triton_kernel_wrapper_functional_proxy_442['DGAMMA'];  triton_kernel_wrapper_functional_proxy_442 = None
        empty_608 = torch.ops.aten.empty.memory_format([512, 1024, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_183 = torch.ops.aten.permute.default(empty_608, [0, 1, 2]);  empty_608 = None
        triton_kernel_wrapper_functional_proxy_443 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 542, constant_args_idx = 754, grid = [(1024, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2357, 'DY': view_2320, 'INVSTD': rsqrt_26, 'GAMMA': primals_160, 'DBETA': getitem_995, 'DGAMMA': getitem_996, 'DX': permute_183, 'M': 100352, 'HW': 196, 'stride_n': 200704, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2357 = view_2320 = rsqrt_26 = primals_160 = permute_183 = None
        getitem_997 = triton_kernel_wrapper_functional_proxy_443['DX'];  triton_kernel_wrapper_functional_proxy_443 = None
        convert_element_type_default_53 = torch.ops.prims.convert_element_type.default(getitem_996, torch.float32);  getitem_996 = None
        convert_element_type_default_52 = torch.ops.prims.convert_element_type.default(getitem_995, torch.float32);  getitem_995 = None
        empty_609 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_444 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 543, constant_args_idx = 755, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_334, 'S_ptr': getitem_335, 'M_ptr': getitem_336, 'Y_ptr': empty_609, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_334 = getitem_335 = getitem_336 = empty_609 = None
        getitem_998 = triton_kernel_wrapper_functional_proxy_444['Y_ptr'];  triton_kernel_wrapper_functional_proxy_444 = None
        view_2373 = torch.ops.aten.view.default(getitem_997, [512, 1024, 14, 14]);  getitem_997 = None
        empty_610 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_53 = torch.ops.aten.expand.default(empty_610, [512, 256, 14, 14]);  empty_610 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(view_2373, expand_53, convert_element_type_27, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_53 = convert_element_type_27 = None
        getitem_999 = convolution_backward_52[0];  convolution_backward_52 = None
        empty_611 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_54 = torch.ops.aten.expand.default(empty_611, [1024, 256, 1, 1]);  empty_611 = None
        view_2374 = torch.ops.aten.view.default(getitem_998, [512, 98, 512]);  getitem_998 = None
        view_2375 = torch.ops.aten.view.default(view_2374, [512, 256, 14, 14]);  view_2374 = None
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(view_2373, view_2375, expand_54, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2373 = view_2375 = expand_54 = None
        getitem_1003 = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_197 = torch.ops.prims.convert_element_type.default(getitem_1003, torch.float32);  getitem_1003 = None
        triton_kernel_wrapper_functional_proxy_445 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 756, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_333, 'Y_ptr': full_default_342, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_333 = full_default_342 = None
        getitem_1005 = triton_kernel_wrapper_functional_proxy_445['Y_ptr'];  triton_kernel_wrapper_functional_proxy_445 = None
        view_2378 = torch.ops.aten.view.default(getitem_1005, [512, 98, 512]);  getitem_1005 = None
        view_2379 = torch.ops.aten.view.default(view_2378, [512, 256, 14, 14]);  view_2378 = None
        mul_396 = torch.ops.aten.mul.Tensor(getitem_999, view_2379);  getitem_999 = view_2379 = None
        empty_612 = torch.ops.aten.empty.memory_format([50176, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_446 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 544, constant_args_idx = 757, grid = [(50176, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_328, 'S_ptr': getitem_329, 'M_ptr': getitem_330, 'Y_ptr': empty_612, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_328 = getitem_329 = getitem_330 = empty_612 = None
        getitem_1006 = triton_kernel_wrapper_functional_proxy_446['Y_ptr'];  triton_kernel_wrapper_functional_proxy_446 = None
        view_2394 = torch.ops.aten.view.default(mul_396, [512, 256, 196]);  mul_396 = None
        view_2395 = torch.ops.aten.view.default(getitem_1006, [512, 98, 512]);  getitem_1006 = None
        view_2396 = torch.ops.aten.view.default(view_2395, [512, 256, 196]);  view_2395 = None
        triton_kernel_wrapper_functional_proxy_447 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 545, constant_args_idx = 758, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2396, 'DY': view_2394, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1007 = triton_kernel_wrapper_functional_proxy_447['DBETA']
        getitem_1008 = triton_kernel_wrapper_functional_proxy_447['DGAMMA'];  triton_kernel_wrapper_functional_proxy_447 = None
        empty_613 = torch.ops.aten.empty.memory_format([512, 256, 196], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_184 = torch.ops.aten.permute.default(empty_613, [0, 1, 2]);  empty_613 = None
        triton_kernel_wrapper_functional_proxy_448 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 546, constant_args_idx = 759, grid = [(256, 98, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2396, 'DY': view_2394, 'INVSTD': rsqrt_25, 'GAMMA': primals_154, 'DBETA': getitem_1007, 'DGAMMA': getitem_1008, 'DX': permute_184, 'M': 100352, 'HW': 196, 'stride_n': 50176, 'stride_c': 196, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2396 = view_2394 = rsqrt_25 = primals_154 = permute_184 = None
        getitem_1009 = triton_kernel_wrapper_functional_proxy_448['DX'];  triton_kernel_wrapper_functional_proxy_448 = None
        convert_element_type_default_51 = torch.ops.prims.convert_element_type.default(getitem_1008, torch.float32);  getitem_1008 = None
        convert_element_type_default_50 = torch.ops.prims.convert_element_type.default(getitem_1007, torch.float32);  getitem_1007 = None
        empty_614 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_449 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 547, constant_args_idx = 760, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_321, 'S_ptr': getitem_322, 'M_ptr': getitem_323, 'Y_ptr': empty_614, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_321 = getitem_322 = getitem_323 = empty_614 = None
        getitem_1010 = triton_kernel_wrapper_functional_proxy_449['Y_ptr'];  triton_kernel_wrapper_functional_proxy_449 = None
        view_2412 = torch.ops.aten.view.default(getitem_1009, [512, 256, 14, 14]);  getitem_1009 = None
        empty_615 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_55 = torch.ops.aten.expand.default(empty_615, [512, 256, 28, 28]);  empty_615 = None
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(view_2412, expand_55, convert_element_type_26, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_55 = convert_element_type_26 = None
        getitem_1011 = convolution_backward_54[0];  convolution_backward_54 = None
        empty_616 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_56 = torch.ops.aten.expand.default(empty_616, [256, 256, 3, 3]);  empty_616 = None
        view_2413 = torch.ops.aten.view.default(getitem_1010, [512, 392, 512]);  getitem_1010 = None
        view_2414 = torch.ops.aten.view.default(view_2413, [512, 256, 28, 28]);  view_2413 = None
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(view_2412, view_2414, expand_56, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2412 = view_2414 = expand_56 = None
        getitem_1015 = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_202 = torch.ops.prims.convert_element_type.default(getitem_1015, torch.float32);  getitem_1015 = None
        triton_kernel_wrapper_functional_proxy_450 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 761, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_320, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_320 = None
        getitem_1017 = triton_kernel_wrapper_functional_proxy_450['Y_ptr'];  triton_kernel_wrapper_functional_proxy_450 = None
        view_2417 = torch.ops.aten.view.default(getitem_1017, [512, 392, 512]);  getitem_1017 = None
        view_2418 = torch.ops.aten.view.default(view_2417, [512, 256, 28, 28]);  view_2417 = None
        mul_397 = torch.ops.aten.mul.Tensor(getitem_1011, view_2418);  getitem_1011 = view_2418 = None
        empty_617 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_451 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 548, constant_args_idx = 762, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_315, 'S_ptr': getitem_316, 'M_ptr': getitem_317, 'Y_ptr': empty_617, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_315 = getitem_316 = getitem_317 = empty_617 = None
        getitem_1018 = triton_kernel_wrapper_functional_proxy_451['Y_ptr'];  triton_kernel_wrapper_functional_proxy_451 = None
        view_2433 = torch.ops.aten.view.default(mul_397, [512, 256, 784]);  mul_397 = None
        view_2434 = torch.ops.aten.view.default(getitem_1018, [512, 392, 512]);  getitem_1018 = None
        view_2435 = torch.ops.aten.view.default(view_2434, [512, 256, 784]);  view_2434 = None
        triton_kernel_wrapper_functional_proxy_452 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 549, constant_args_idx = 763, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2435, 'DY': view_2433, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1019 = triton_kernel_wrapper_functional_proxy_452['DBETA']
        getitem_1020 = triton_kernel_wrapper_functional_proxy_452['DGAMMA'];  triton_kernel_wrapper_functional_proxy_452 = None
        empty_618 = torch.ops.aten.empty.memory_format([512, 256, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_185 = torch.ops.aten.permute.default(empty_618, [0, 1, 2]);  empty_618 = None
        triton_kernel_wrapper_functional_proxy_453 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 550, constant_args_idx = 764, grid = [(256, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2435, 'DY': view_2433, 'INVSTD': rsqrt_24, 'GAMMA': primals_148, 'DBETA': getitem_1019, 'DGAMMA': getitem_1020, 'DX': permute_185, 'M': 401408, 'HW': 784, 'stride_n': 200704, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2435 = view_2433 = rsqrt_24 = primals_148 = permute_185 = None
        getitem_1021 = triton_kernel_wrapper_functional_proxy_453['DX'];  triton_kernel_wrapper_functional_proxy_453 = None
        convert_element_type_default_49 = torch.ops.prims.convert_element_type.default(getitem_1020, torch.float32);  getitem_1020 = None
        convert_element_type_default_48 = torch.ops.prims.convert_element_type.default(getitem_1019, torch.float32);  getitem_1019 = None
        empty_619 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_454 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 551, constant_args_idx = 765, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_308, 'S_ptr': getitem_309, 'M_ptr': getitem_310, 'Y_ptr': empty_619, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_308 = getitem_309 = getitem_310 = empty_619 = None
        getitem_1022 = triton_kernel_wrapper_functional_proxy_454['Y_ptr'];  triton_kernel_wrapper_functional_proxy_454 = None
        view_2451 = torch.ops.aten.view.default(getitem_1021, [512, 256, 28, 28]);  getitem_1021 = None
        empty_620 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_57 = torch.ops.aten.expand.default(empty_620, [512, 512, 28, 28]);  empty_620 = None
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(view_2451, expand_57, convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_57 = convert_element_type_25 = None
        getitem_1023 = convolution_backward_56[0];  convolution_backward_56 = None
        empty_621 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_58 = torch.ops.aten.expand.default(empty_621, [256, 512, 1, 1]);  empty_621 = None
        view_2452 = torch.ops.aten.view.default(getitem_1022, [512, 784, 512]);  getitem_1022 = None
        view_2453 = torch.ops.aten.view.default(view_2452, [512, 512, 28, 28]);  view_2452 = None
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(view_2451, view_2453, expand_58, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2451 = view_2453 = expand_58 = None
        getitem_1027 = convolution_backward_57[1];  convolution_backward_57 = None
        convert_element_type_207 = torch.ops.prims.convert_element_type.default(getitem_1027, torch.float32);  getitem_1027 = None
        add_236 = torch.ops.aten.add.Tensor(getitem_988, getitem_1023);  getitem_988 = getitem_1023 = None
        full_default_395 = torch.ops.aten.full.default([401408, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_455 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 766, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_307, 'Y_ptr': full_default_395, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_307 = None
        getitem_1029 = triton_kernel_wrapper_functional_proxy_455['Y_ptr'];  triton_kernel_wrapper_functional_proxy_455 = None
        view_2456 = torch.ops.aten.view.default(getitem_1029, [512, 784, 512]);  getitem_1029 = None
        view_2457 = torch.ops.aten.view.default(view_2456, [512, 512, 28, 28]);  view_2456 = None
        mul_398 = torch.ops.aten.mul.Tensor(add_236, view_2457);  add_236 = view_2457 = None
        empty_622 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_456 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 552, constant_args_idx = 767, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_302, 'S_ptr': getitem_303, 'M_ptr': getitem_304, 'Y_ptr': empty_622, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_302 = getitem_303 = getitem_304 = empty_622 = None
        getitem_1030 = triton_kernel_wrapper_functional_proxy_456['Y_ptr'];  triton_kernel_wrapper_functional_proxy_456 = None
        view_2472 = torch.ops.aten.view.default(mul_398, [512, 512, 784])
        view_2473 = torch.ops.aten.view.default(getitem_1030, [512, 784, 512]);  getitem_1030 = None
        view_2474 = torch.ops.aten.view.default(view_2473, [512, 512, 784]);  view_2473 = None
        triton_kernel_wrapper_functional_proxy_457 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 553, constant_args_idx = 768, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2474, 'DY': view_2472, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1031 = triton_kernel_wrapper_functional_proxy_457['DBETA']
        getitem_1032 = triton_kernel_wrapper_functional_proxy_457['DGAMMA'];  triton_kernel_wrapper_functional_proxy_457 = None
        empty_623 = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_186 = torch.ops.aten.permute.default(empty_623, [0, 1, 2]);  empty_623 = None
        triton_kernel_wrapper_functional_proxy_458 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 554, constant_args_idx = 769, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2474, 'DY': view_2472, 'INVSTD': rsqrt_23, 'GAMMA': primals_142, 'DBETA': getitem_1031, 'DGAMMA': getitem_1032, 'DX': permute_186, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2474 = view_2472 = rsqrt_23 = primals_142 = permute_186 = None
        getitem_1033 = triton_kernel_wrapper_functional_proxy_458['DX'];  triton_kernel_wrapper_functional_proxy_458 = None
        convert_element_type_default_47 = torch.ops.prims.convert_element_type.default(getitem_1032, torch.float32);  getitem_1032 = None
        convert_element_type_default_46 = torch.ops.prims.convert_element_type.default(getitem_1031, torch.float32);  getitem_1031 = None
        empty_624 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_459 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 555, constant_args_idx = 770, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_295, 'S_ptr': getitem_296, 'M_ptr': getitem_297, 'Y_ptr': empty_624, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_295 = getitem_296 = getitem_297 = empty_624 = None
        getitem_1034 = triton_kernel_wrapper_functional_proxy_459['Y_ptr'];  triton_kernel_wrapper_functional_proxy_459 = None
        view_2490 = torch.ops.aten.view.default(getitem_1033, [512, 512, 28, 28]);  getitem_1033 = None
        empty_625 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_59 = torch.ops.aten.expand.default(empty_625, [512, 128, 28, 28]);  empty_625 = None
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(view_2490, expand_59, convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_59 = convert_element_type_24 = None
        getitem_1035 = convolution_backward_58[0];  convolution_backward_58 = None
        empty_626 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_60 = torch.ops.aten.expand.default(empty_626, [512, 128, 1, 1]);  empty_626 = None
        view_2491 = torch.ops.aten.view.default(getitem_1034, [512, 196, 512]);  getitem_1034 = None
        view_2492 = torch.ops.aten.view.default(view_2491, [512, 128, 28, 28]);  view_2491 = None
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(view_2490, view_2492, expand_60, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2490 = view_2492 = expand_60 = None
        getitem_1039 = convolution_backward_59[1];  convolution_backward_59 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(getitem_1039, torch.float32);  getitem_1039 = None
        triton_kernel_wrapper_functional_proxy_460 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 771, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_294, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_294 = None
        getitem_1041 = triton_kernel_wrapper_functional_proxy_460['Y_ptr'];  triton_kernel_wrapper_functional_proxy_460 = None
        view_2495 = torch.ops.aten.view.default(getitem_1041, [512, 196, 512]);  getitem_1041 = None
        view_2496 = torch.ops.aten.view.default(view_2495, [512, 128, 28, 28]);  view_2495 = None
        mul_399 = torch.ops.aten.mul.Tensor(getitem_1035, view_2496);  getitem_1035 = view_2496 = None
        empty_627 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_461 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 556, constant_args_idx = 772, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_289, 'S_ptr': getitem_290, 'M_ptr': getitem_291, 'Y_ptr': empty_627, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_289 = getitem_290 = getitem_291 = empty_627 = None
        getitem_1042 = triton_kernel_wrapper_functional_proxy_461['Y_ptr'];  triton_kernel_wrapper_functional_proxy_461 = None
        view_2511 = torch.ops.aten.view.default(mul_399, [512, 128, 784]);  mul_399 = None
        view_2512 = torch.ops.aten.view.default(getitem_1042, [512, 196, 512]);  getitem_1042 = None
        view_2513 = torch.ops.aten.view.default(view_2512, [512, 128, 784]);  view_2512 = None
        full_default_64 = torch.ops.aten.full.default([128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_462 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 557, constant_args_idx = 773, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2513, 'DY': view_2511, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1043 = triton_kernel_wrapper_functional_proxy_462['DBETA']
        getitem_1044 = triton_kernel_wrapper_functional_proxy_462['DGAMMA'];  triton_kernel_wrapper_functional_proxy_462 = None
        empty_628 = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_187 = torch.ops.aten.permute.default(empty_628, [0, 1, 2]);  empty_628 = None
        triton_kernel_wrapper_functional_proxy_463 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 558, constant_args_idx = 774, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2513, 'DY': view_2511, 'INVSTD': rsqrt_22, 'GAMMA': primals_136, 'DBETA': getitem_1043, 'DGAMMA': getitem_1044, 'DX': permute_187, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2513 = view_2511 = rsqrt_22 = primals_136 = permute_187 = None
        getitem_1045 = triton_kernel_wrapper_functional_proxy_463['DX'];  triton_kernel_wrapper_functional_proxy_463 = None
        convert_element_type_default_45 = torch.ops.prims.convert_element_type.default(getitem_1044, torch.float32);  getitem_1044 = None
        convert_element_type_default_44 = torch.ops.prims.convert_element_type.default(getitem_1043, torch.float32);  getitem_1043 = None
        empty_629 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_464 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 559, constant_args_idx = 775, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_282, 'S_ptr': getitem_283, 'M_ptr': getitem_284, 'Y_ptr': empty_629, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_282 = getitem_283 = getitem_284 = empty_629 = None
        getitem_1046 = triton_kernel_wrapper_functional_proxy_464['Y_ptr'];  triton_kernel_wrapper_functional_proxy_464 = None
        view_2529 = torch.ops.aten.view.default(getitem_1045, [512, 128, 28, 28]);  getitem_1045 = None
        empty_630 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_61 = torch.ops.aten.expand.default(empty_630, [512, 128, 28, 28]);  empty_630 = None
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(view_2529, expand_61, convert_element_type_23, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_61 = convert_element_type_23 = None
        getitem_1047 = convolution_backward_60[0];  convolution_backward_60 = None
        empty_631 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_62 = torch.ops.aten.expand.default(empty_631, [128, 128, 3, 3]);  empty_631 = None
        view_2530 = torch.ops.aten.view.default(getitem_1046, [512, 196, 512]);  getitem_1046 = None
        view_2531 = torch.ops.aten.view.default(view_2530, [512, 128, 28, 28]);  view_2530 = None
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(view_2529, view_2531, expand_62, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2529 = view_2531 = expand_62 = None
        getitem_1051 = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_217 = torch.ops.prims.convert_element_type.default(getitem_1051, torch.float32);  getitem_1051 = None
        triton_kernel_wrapper_functional_proxy_465 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 776, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_281, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_281 = None
        getitem_1053 = triton_kernel_wrapper_functional_proxy_465['Y_ptr'];  triton_kernel_wrapper_functional_proxy_465 = None
        view_2534 = torch.ops.aten.view.default(getitem_1053, [512, 196, 512]);  getitem_1053 = None
        view_2535 = torch.ops.aten.view.default(view_2534, [512, 128, 28, 28]);  view_2534 = None
        mul_400 = torch.ops.aten.mul.Tensor(getitem_1047, view_2535);  getitem_1047 = view_2535 = None
        empty_632 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_466 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 560, constant_args_idx = 777, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_276, 'S_ptr': getitem_277, 'M_ptr': getitem_278, 'Y_ptr': empty_632, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_276 = getitem_277 = getitem_278 = empty_632 = None
        getitem_1054 = triton_kernel_wrapper_functional_proxy_466['Y_ptr'];  triton_kernel_wrapper_functional_proxy_466 = None
        view_2550 = torch.ops.aten.view.default(mul_400, [512, 128, 784]);  mul_400 = None
        view_2551 = torch.ops.aten.view.default(getitem_1054, [512, 196, 512]);  getitem_1054 = None
        view_2552 = torch.ops.aten.view.default(view_2551, [512, 128, 784]);  view_2551 = None
        triton_kernel_wrapper_functional_proxy_467 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 561, constant_args_idx = 778, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2552, 'DY': view_2550, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1055 = triton_kernel_wrapper_functional_proxy_467['DBETA']
        getitem_1056 = triton_kernel_wrapper_functional_proxy_467['DGAMMA'];  triton_kernel_wrapper_functional_proxy_467 = None
        empty_633 = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_188 = torch.ops.aten.permute.default(empty_633, [0, 1, 2]);  empty_633 = None
        triton_kernel_wrapper_functional_proxy_468 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 562, constant_args_idx = 779, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2552, 'DY': view_2550, 'INVSTD': rsqrt_21, 'GAMMA': primals_130, 'DBETA': getitem_1055, 'DGAMMA': getitem_1056, 'DX': permute_188, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2552 = view_2550 = rsqrt_21 = primals_130 = permute_188 = None
        getitem_1057 = triton_kernel_wrapper_functional_proxy_468['DX'];  triton_kernel_wrapper_functional_proxy_468 = None
        convert_element_type_default_43 = torch.ops.prims.convert_element_type.default(getitem_1056, torch.float32);  getitem_1056 = None
        convert_element_type_default_42 = torch.ops.prims.convert_element_type.default(getitem_1055, torch.float32);  getitem_1055 = None
        empty_634 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_469 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 563, constant_args_idx = 780, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_269, 'S_ptr': getitem_270, 'M_ptr': getitem_271, 'Y_ptr': empty_634, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_269 = getitem_270 = getitem_271 = empty_634 = None
        getitem_1058 = triton_kernel_wrapper_functional_proxy_469['Y_ptr'];  triton_kernel_wrapper_functional_proxy_469 = None
        view_2568 = torch.ops.aten.view.default(getitem_1057, [512, 128, 28, 28]);  getitem_1057 = None
        empty_635 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_63 = torch.ops.aten.expand.default(empty_635, [512, 512, 28, 28]);  empty_635 = None
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(view_2568, expand_63, convert_element_type_22, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_63 = convert_element_type_22 = None
        getitem_1059 = convolution_backward_62[0];  convolution_backward_62 = None
        empty_636 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_64 = torch.ops.aten.expand.default(empty_636, [128, 512, 1, 1]);  empty_636 = None
        view_2569 = torch.ops.aten.view.default(getitem_1058, [512, 784, 512]);  getitem_1058 = None
        view_2570 = torch.ops.aten.view.default(view_2569, [512, 512, 28, 28]);  view_2569 = None
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(view_2568, view_2570, expand_64, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2568 = view_2570 = expand_64 = None
        getitem_1063 = convolution_backward_63[1];  convolution_backward_63 = None
        convert_element_type_222 = torch.ops.prims.convert_element_type.default(getitem_1063, torch.float32);  getitem_1063 = None
        add_237 = torch.ops.aten.add.Tensor(mul_398, getitem_1059);  mul_398 = getitem_1059 = None
        triton_kernel_wrapper_functional_proxy_470 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 781, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_268, 'Y_ptr': full_default_395, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_268 = None
        getitem_1065 = triton_kernel_wrapper_functional_proxy_470['Y_ptr'];  triton_kernel_wrapper_functional_proxy_470 = None
        view_2573 = torch.ops.aten.view.default(getitem_1065, [512, 784, 512]);  getitem_1065 = None
        view_2574 = torch.ops.aten.view.default(view_2573, [512, 512, 28, 28]);  view_2573 = None
        mul_401 = torch.ops.aten.mul.Tensor(add_237, view_2574);  add_237 = view_2574 = None
        empty_637 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_471 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 564, constant_args_idx = 782, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_263, 'S_ptr': getitem_264, 'M_ptr': getitem_265, 'Y_ptr': empty_637, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_263 = getitem_264 = getitem_265 = empty_637 = None
        getitem_1066 = triton_kernel_wrapper_functional_proxy_471['Y_ptr'];  triton_kernel_wrapper_functional_proxy_471 = None
        view_2589 = torch.ops.aten.view.default(mul_401, [512, 512, 784])
        view_2590 = torch.ops.aten.view.default(getitem_1066, [512, 784, 512]);  getitem_1066 = None
        view_2591 = torch.ops.aten.view.default(view_2590, [512, 512, 784]);  view_2590 = None
        triton_kernel_wrapper_functional_proxy_472 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 565, constant_args_idx = 783, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2591, 'DY': view_2589, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1067 = triton_kernel_wrapper_functional_proxy_472['DBETA']
        getitem_1068 = triton_kernel_wrapper_functional_proxy_472['DGAMMA'];  triton_kernel_wrapper_functional_proxy_472 = None
        empty_638 = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_189 = torch.ops.aten.permute.default(empty_638, [0, 1, 2]);  empty_638 = None
        triton_kernel_wrapper_functional_proxy_473 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 566, constant_args_idx = 784, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2591, 'DY': view_2589, 'INVSTD': rsqrt_20, 'GAMMA': primals_124, 'DBETA': getitem_1067, 'DGAMMA': getitem_1068, 'DX': permute_189, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2591 = view_2589 = rsqrt_20 = primals_124 = permute_189 = None
        getitem_1069 = triton_kernel_wrapper_functional_proxy_473['DX'];  triton_kernel_wrapper_functional_proxy_473 = None
        convert_element_type_default_41 = torch.ops.prims.convert_element_type.default(getitem_1068, torch.float32);  getitem_1068 = None
        convert_element_type_default_40 = torch.ops.prims.convert_element_type.default(getitem_1067, torch.float32);  getitem_1067 = None
        empty_639 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_474 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 567, constant_args_idx = 785, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_256, 'S_ptr': getitem_257, 'M_ptr': getitem_258, 'Y_ptr': empty_639, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_256 = getitem_257 = getitem_258 = empty_639 = None
        getitem_1070 = triton_kernel_wrapper_functional_proxy_474['Y_ptr'];  triton_kernel_wrapper_functional_proxy_474 = None
        view_2607 = torch.ops.aten.view.default(getitem_1069, [512, 512, 28, 28]);  getitem_1069 = None
        empty_640 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_65 = torch.ops.aten.expand.default(empty_640, [512, 128, 28, 28]);  empty_640 = None
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(view_2607, expand_65, convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_65 = convert_element_type_21 = None
        getitem_1071 = convolution_backward_64[0];  convolution_backward_64 = None
        empty_641 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_66 = torch.ops.aten.expand.default(empty_641, [512, 128, 1, 1]);  empty_641 = None
        view_2608 = torch.ops.aten.view.default(getitem_1070, [512, 196, 512]);  getitem_1070 = None
        view_2609 = torch.ops.aten.view.default(view_2608, [512, 128, 28, 28]);  view_2608 = None
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(view_2607, view_2609, expand_66, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2607 = view_2609 = expand_66 = None
        getitem_1075 = convolution_backward_65[1];  convolution_backward_65 = None
        convert_element_type_227 = torch.ops.prims.convert_element_type.default(getitem_1075, torch.float32);  getitem_1075 = None
        triton_kernel_wrapper_functional_proxy_475 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 786, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_255, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_255 = None
        getitem_1077 = triton_kernel_wrapper_functional_proxy_475['Y_ptr'];  triton_kernel_wrapper_functional_proxy_475 = None
        view_2612 = torch.ops.aten.view.default(getitem_1077, [512, 196, 512]);  getitem_1077 = None
        view_2613 = torch.ops.aten.view.default(view_2612, [512, 128, 28, 28]);  view_2612 = None
        mul_402 = torch.ops.aten.mul.Tensor(getitem_1071, view_2613);  getitem_1071 = view_2613 = None
        empty_642 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_476 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 568, constant_args_idx = 787, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_250, 'S_ptr': getitem_251, 'M_ptr': getitem_252, 'Y_ptr': empty_642, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_250 = getitem_251 = getitem_252 = empty_642 = None
        getitem_1078 = triton_kernel_wrapper_functional_proxy_476['Y_ptr'];  triton_kernel_wrapper_functional_proxy_476 = None
        view_2628 = torch.ops.aten.view.default(mul_402, [512, 128, 784]);  mul_402 = None
        view_2629 = torch.ops.aten.view.default(getitem_1078, [512, 196, 512]);  getitem_1078 = None
        view_2630 = torch.ops.aten.view.default(view_2629, [512, 128, 784]);  view_2629 = None
        triton_kernel_wrapper_functional_proxy_477 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 569, constant_args_idx = 788, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2630, 'DY': view_2628, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1079 = triton_kernel_wrapper_functional_proxy_477['DBETA']
        getitem_1080 = triton_kernel_wrapper_functional_proxy_477['DGAMMA'];  triton_kernel_wrapper_functional_proxy_477 = None
        empty_643 = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_190 = torch.ops.aten.permute.default(empty_643, [0, 1, 2]);  empty_643 = None
        triton_kernel_wrapper_functional_proxy_478 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 570, constant_args_idx = 789, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2630, 'DY': view_2628, 'INVSTD': rsqrt_19, 'GAMMA': primals_118, 'DBETA': getitem_1079, 'DGAMMA': getitem_1080, 'DX': permute_190, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2630 = view_2628 = rsqrt_19 = primals_118 = permute_190 = None
        getitem_1081 = triton_kernel_wrapper_functional_proxy_478['DX'];  triton_kernel_wrapper_functional_proxy_478 = None
        convert_element_type_default_39 = torch.ops.prims.convert_element_type.default(getitem_1080, torch.float32);  getitem_1080 = None
        convert_element_type_default_38 = torch.ops.prims.convert_element_type.default(getitem_1079, torch.float32);  getitem_1079 = None
        empty_644 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_479 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 571, constant_args_idx = 790, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_243, 'S_ptr': getitem_244, 'M_ptr': getitem_245, 'Y_ptr': empty_644, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_243 = getitem_244 = getitem_245 = empty_644 = None
        getitem_1082 = triton_kernel_wrapper_functional_proxy_479['Y_ptr'];  triton_kernel_wrapper_functional_proxy_479 = None
        view_2646 = torch.ops.aten.view.default(getitem_1081, [512, 128, 28, 28]);  getitem_1081 = None
        empty_645 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_67 = torch.ops.aten.expand.default(empty_645, [512, 128, 28, 28]);  empty_645 = None
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(view_2646, expand_67, convert_element_type_20, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_67 = convert_element_type_20 = None
        getitem_1083 = convolution_backward_66[0];  convolution_backward_66 = None
        empty_646 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_68 = torch.ops.aten.expand.default(empty_646, [128, 128, 3, 3]);  empty_646 = None
        view_2647 = torch.ops.aten.view.default(getitem_1082, [512, 196, 512]);  getitem_1082 = None
        view_2648 = torch.ops.aten.view.default(view_2647, [512, 128, 28, 28]);  view_2647 = None
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(view_2646, view_2648, expand_68, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2646 = view_2648 = expand_68 = None
        getitem_1087 = convolution_backward_67[1];  convolution_backward_67 = None
        convert_element_type_232 = torch.ops.prims.convert_element_type.default(getitem_1087, torch.float32);  getitem_1087 = None
        triton_kernel_wrapper_functional_proxy_480 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 791, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_242, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_242 = None
        getitem_1089 = triton_kernel_wrapper_functional_proxy_480['Y_ptr'];  triton_kernel_wrapper_functional_proxy_480 = None
        view_2651 = torch.ops.aten.view.default(getitem_1089, [512, 196, 512]);  getitem_1089 = None
        view_2652 = torch.ops.aten.view.default(view_2651, [512, 128, 28, 28]);  view_2651 = None
        mul_403 = torch.ops.aten.mul.Tensor(getitem_1083, view_2652);  getitem_1083 = view_2652 = None
        empty_647 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_481 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 572, constant_args_idx = 792, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_237, 'S_ptr': getitem_238, 'M_ptr': getitem_239, 'Y_ptr': empty_647, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_237 = getitem_238 = getitem_239 = empty_647 = None
        getitem_1090 = triton_kernel_wrapper_functional_proxy_481['Y_ptr'];  triton_kernel_wrapper_functional_proxy_481 = None
        view_2667 = torch.ops.aten.view.default(mul_403, [512, 128, 784]);  mul_403 = None
        view_2668 = torch.ops.aten.view.default(getitem_1090, [512, 196, 512]);  getitem_1090 = None
        view_2669 = torch.ops.aten.view.default(view_2668, [512, 128, 784]);  view_2668 = None
        triton_kernel_wrapper_functional_proxy_482 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 573, constant_args_idx = 793, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2669, 'DY': view_2667, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1091 = triton_kernel_wrapper_functional_proxy_482['DBETA']
        getitem_1092 = triton_kernel_wrapper_functional_proxy_482['DGAMMA'];  triton_kernel_wrapper_functional_proxy_482 = None
        empty_648 = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_191 = torch.ops.aten.permute.default(empty_648, [0, 1, 2]);  empty_648 = None
        triton_kernel_wrapper_functional_proxy_483 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 574, constant_args_idx = 794, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2669, 'DY': view_2667, 'INVSTD': rsqrt_18, 'GAMMA': primals_112, 'DBETA': getitem_1091, 'DGAMMA': getitem_1092, 'DX': permute_191, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2669 = view_2667 = rsqrt_18 = primals_112 = permute_191 = None
        getitem_1093 = triton_kernel_wrapper_functional_proxy_483['DX'];  triton_kernel_wrapper_functional_proxy_483 = None
        convert_element_type_default_37 = torch.ops.prims.convert_element_type.default(getitem_1092, torch.float32);  getitem_1092 = None
        convert_element_type_default_36 = torch.ops.prims.convert_element_type.default(getitem_1091, torch.float32);  getitem_1091 = None
        empty_649 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_484 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 575, constant_args_idx = 795, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_230, 'S_ptr': getitem_231, 'M_ptr': getitem_232, 'Y_ptr': empty_649, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_230 = getitem_231 = getitem_232 = empty_649 = None
        getitem_1094 = triton_kernel_wrapper_functional_proxy_484['Y_ptr'];  triton_kernel_wrapper_functional_proxy_484 = None
        view_2685 = torch.ops.aten.view.default(getitem_1093, [512, 128, 28, 28]);  getitem_1093 = None
        empty_650 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_69 = torch.ops.aten.expand.default(empty_650, [512, 512, 28, 28]);  empty_650 = None
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(view_2685, expand_69, convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_69 = convert_element_type_19 = None
        getitem_1095 = convolution_backward_68[0];  convolution_backward_68 = None
        empty_651 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_70 = torch.ops.aten.expand.default(empty_651, [128, 512, 1, 1]);  empty_651 = None
        view_2686 = torch.ops.aten.view.default(getitem_1094, [512, 784, 512]);  getitem_1094 = None
        view_2687 = torch.ops.aten.view.default(view_2686, [512, 512, 28, 28]);  view_2686 = None
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(view_2685, view_2687, expand_70, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2685 = view_2687 = expand_70 = None
        getitem_1099 = convolution_backward_69[1];  convolution_backward_69 = None
        convert_element_type_237 = torch.ops.prims.convert_element_type.default(getitem_1099, torch.float32);  getitem_1099 = None
        add_238 = torch.ops.aten.add.Tensor(mul_401, getitem_1095);  mul_401 = getitem_1095 = None
        triton_kernel_wrapper_functional_proxy_485 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 796, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_229, 'Y_ptr': full_default_395, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_229 = None
        getitem_1101 = triton_kernel_wrapper_functional_proxy_485['Y_ptr'];  triton_kernel_wrapper_functional_proxy_485 = None
        view_2690 = torch.ops.aten.view.default(getitem_1101, [512, 784, 512]);  getitem_1101 = None
        view_2691 = torch.ops.aten.view.default(view_2690, [512, 512, 28, 28]);  view_2690 = None
        mul_404 = torch.ops.aten.mul.Tensor(add_238, view_2691);  add_238 = view_2691 = None
        empty_652 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_486 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 576, constant_args_idx = 797, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_224, 'S_ptr': getitem_225, 'M_ptr': getitem_226, 'Y_ptr': empty_652, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_224 = getitem_225 = getitem_226 = empty_652 = None
        getitem_1102 = triton_kernel_wrapper_functional_proxy_486['Y_ptr'];  triton_kernel_wrapper_functional_proxy_486 = None
        view_2706 = torch.ops.aten.view.default(mul_404, [512, 512, 784])
        view_2707 = torch.ops.aten.view.default(getitem_1102, [512, 784, 512]);  getitem_1102 = None
        view_2708 = torch.ops.aten.view.default(view_2707, [512, 512, 784]);  view_2707 = None
        triton_kernel_wrapper_functional_proxy_487 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 577, constant_args_idx = 798, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2708, 'DY': view_2706, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1103 = triton_kernel_wrapper_functional_proxy_487['DBETA']
        getitem_1104 = triton_kernel_wrapper_functional_proxy_487['DGAMMA'];  triton_kernel_wrapper_functional_proxy_487 = None
        empty_653 = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_192 = torch.ops.aten.permute.default(empty_653, [0, 1, 2]);  empty_653 = None
        triton_kernel_wrapper_functional_proxy_488 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 578, constant_args_idx = 799, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2708, 'DY': view_2706, 'INVSTD': rsqrt_17, 'GAMMA': primals_106, 'DBETA': getitem_1103, 'DGAMMA': getitem_1104, 'DX': permute_192, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2708 = view_2706 = rsqrt_17 = primals_106 = permute_192 = None
        getitem_1105 = triton_kernel_wrapper_functional_proxy_488['DX'];  triton_kernel_wrapper_functional_proxy_488 = None
        convert_element_type_default_35 = torch.ops.prims.convert_element_type.default(getitem_1104, torch.float32);  getitem_1104 = None
        convert_element_type_default_34 = torch.ops.prims.convert_element_type.default(getitem_1103, torch.float32);  getitem_1103 = None
        empty_654 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_489 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 579, constant_args_idx = 800, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_217, 'S_ptr': getitem_218, 'M_ptr': getitem_219, 'Y_ptr': empty_654, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_217 = getitem_218 = getitem_219 = empty_654 = None
        getitem_1106 = triton_kernel_wrapper_functional_proxy_489['Y_ptr'];  triton_kernel_wrapper_functional_proxy_489 = None
        view_2724 = torch.ops.aten.view.default(getitem_1105, [512, 512, 28, 28]);  getitem_1105 = None
        empty_655 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_71 = torch.ops.aten.expand.default(empty_655, [512, 128, 28, 28]);  empty_655 = None
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(view_2724, expand_71, convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_71 = convert_element_type_18 = None
        getitem_1107 = convolution_backward_70[0];  convolution_backward_70 = None
        empty_656 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_72 = torch.ops.aten.expand.default(empty_656, [512, 128, 1, 1]);  empty_656 = None
        view_2725 = torch.ops.aten.view.default(getitem_1106, [512, 196, 512]);  getitem_1106 = None
        view_2726 = torch.ops.aten.view.default(view_2725, [512, 128, 28, 28]);  view_2725 = None
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(view_2724, view_2726, expand_72, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2724 = view_2726 = expand_72 = None
        getitem_1111 = convolution_backward_71[1];  convolution_backward_71 = None
        convert_element_type_242 = torch.ops.prims.convert_element_type.default(getitem_1111, torch.float32);  getitem_1111 = None
        triton_kernel_wrapper_functional_proxy_490 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 801, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_216, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_216 = None
        getitem_1113 = triton_kernel_wrapper_functional_proxy_490['Y_ptr'];  triton_kernel_wrapper_functional_proxy_490 = None
        view_2729 = torch.ops.aten.view.default(getitem_1113, [512, 196, 512]);  getitem_1113 = None
        view_2730 = torch.ops.aten.view.default(view_2729, [512, 128, 28, 28]);  view_2729 = None
        mul_405 = torch.ops.aten.mul.Tensor(getitem_1107, view_2730);  getitem_1107 = view_2730 = None
        empty_657 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_491 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 580, constant_args_idx = 802, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_211, 'S_ptr': getitem_212, 'M_ptr': getitem_213, 'Y_ptr': empty_657, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_211 = getitem_212 = getitem_213 = empty_657 = None
        getitem_1114 = triton_kernel_wrapper_functional_proxy_491['Y_ptr'];  triton_kernel_wrapper_functional_proxy_491 = None
        view_2745 = torch.ops.aten.view.default(mul_405, [512, 128, 784]);  mul_405 = None
        view_2746 = torch.ops.aten.view.default(getitem_1114, [512, 196, 512]);  getitem_1114 = None
        view_2747 = torch.ops.aten.view.default(view_2746, [512, 128, 784]);  view_2746 = None
        triton_kernel_wrapper_functional_proxy_492 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 581, constant_args_idx = 803, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2747, 'DY': view_2745, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1115 = triton_kernel_wrapper_functional_proxy_492['DBETA']
        getitem_1116 = triton_kernel_wrapper_functional_proxy_492['DGAMMA'];  triton_kernel_wrapper_functional_proxy_492 = None
        empty_658 = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_193 = torch.ops.aten.permute.default(empty_658, [0, 1, 2]);  empty_658 = None
        triton_kernel_wrapper_functional_proxy_493 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 582, constant_args_idx = 804, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2747, 'DY': view_2745, 'INVSTD': rsqrt_16, 'GAMMA': primals_100, 'DBETA': getitem_1115, 'DGAMMA': getitem_1116, 'DX': permute_193, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2747 = view_2745 = rsqrt_16 = primals_100 = permute_193 = None
        getitem_1117 = triton_kernel_wrapper_functional_proxy_493['DX'];  triton_kernel_wrapper_functional_proxy_493 = None
        convert_element_type_default_33 = torch.ops.prims.convert_element_type.default(getitem_1116, torch.float32);  getitem_1116 = None
        convert_element_type_default_32 = torch.ops.prims.convert_element_type.default(getitem_1115, torch.float32);  getitem_1115 = None
        empty_659 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_494 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 583, constant_args_idx = 805, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_204, 'S_ptr': getitem_205, 'M_ptr': getitem_206, 'Y_ptr': empty_659, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_204 = getitem_205 = getitem_206 = empty_659 = None
        getitem_1118 = triton_kernel_wrapper_functional_proxy_494['Y_ptr'];  triton_kernel_wrapper_functional_proxy_494 = None
        view_2763 = torch.ops.aten.view.default(getitem_1117, [512, 128, 28, 28]);  getitem_1117 = None
        empty_660 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_73 = torch.ops.aten.expand.default(empty_660, [512, 128, 28, 28]);  empty_660 = None
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(view_2763, expand_73, convert_element_type_17, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_73 = convert_element_type_17 = None
        getitem_1119 = convolution_backward_72[0];  convolution_backward_72 = None
        empty_661 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_74 = torch.ops.aten.expand.default(empty_661, [128, 128, 3, 3]);  empty_661 = None
        view_2764 = torch.ops.aten.view.default(getitem_1118, [512, 196, 512]);  getitem_1118 = None
        view_2765 = torch.ops.aten.view.default(view_2764, [512, 128, 28, 28]);  view_2764 = None
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(view_2763, view_2765, expand_74, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2763 = view_2765 = expand_74 = None
        getitem_1123 = convolution_backward_73[1];  convolution_backward_73 = None
        convert_element_type_247 = torch.ops.prims.convert_element_type.default(getitem_1123, torch.float32);  getitem_1123 = None
        triton_kernel_wrapper_functional_proxy_495 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 806, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_203, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_203 = None
        getitem_1125 = triton_kernel_wrapper_functional_proxy_495['Y_ptr'];  triton_kernel_wrapper_functional_proxy_495 = None
        view_2768 = torch.ops.aten.view.default(getitem_1125, [512, 196, 512]);  getitem_1125 = None
        view_2769 = torch.ops.aten.view.default(view_2768, [512, 128, 28, 28]);  view_2768 = None
        mul_406 = torch.ops.aten.mul.Tensor(getitem_1119, view_2769);  getitem_1119 = view_2769 = None
        empty_662 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_496 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 584, constant_args_idx = 807, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_198, 'S_ptr': getitem_199, 'M_ptr': getitem_200, 'Y_ptr': empty_662, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_198 = getitem_199 = getitem_200 = empty_662 = None
        getitem_1126 = triton_kernel_wrapper_functional_proxy_496['Y_ptr'];  triton_kernel_wrapper_functional_proxy_496 = None
        view_2784 = torch.ops.aten.view.default(mul_406, [512, 128, 784]);  mul_406 = None
        view_2785 = torch.ops.aten.view.default(getitem_1126, [512, 196, 512]);  getitem_1126 = None
        view_2786 = torch.ops.aten.view.default(view_2785, [512, 128, 784]);  view_2785 = None
        triton_kernel_wrapper_functional_proxy_497 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 585, constant_args_idx = 808, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2786, 'DY': view_2784, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1127 = triton_kernel_wrapper_functional_proxy_497['DBETA']
        getitem_1128 = triton_kernel_wrapper_functional_proxy_497['DGAMMA'];  triton_kernel_wrapper_functional_proxy_497 = None
        empty_663 = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_194 = torch.ops.aten.permute.default(empty_663, [0, 1, 2]);  empty_663 = None
        triton_kernel_wrapper_functional_proxy_498 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 586, constant_args_idx = 809, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2786, 'DY': view_2784, 'INVSTD': rsqrt_15, 'GAMMA': primals_94, 'DBETA': getitem_1127, 'DGAMMA': getitem_1128, 'DX': permute_194, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2786 = view_2784 = rsqrt_15 = primals_94 = permute_194 = None
        getitem_1129 = triton_kernel_wrapper_functional_proxy_498['DX'];  triton_kernel_wrapper_functional_proxy_498 = None
        convert_element_type_default_31 = torch.ops.prims.convert_element_type.default(getitem_1128, torch.float32);  getitem_1128 = None
        convert_element_type_default_30 = torch.ops.prims.convert_element_type.default(getitem_1127, torch.float32);  getitem_1127 = None
        empty_664 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_499 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 587, constant_args_idx = 810, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_191, 'S_ptr': getitem_192, 'M_ptr': getitem_193, 'Y_ptr': empty_664, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_191 = getitem_192 = getitem_193 = empty_664 = None
        getitem_1130 = triton_kernel_wrapper_functional_proxy_499['Y_ptr'];  triton_kernel_wrapper_functional_proxy_499 = None
        view_2802 = torch.ops.aten.view.default(getitem_1129, [512, 128, 28, 28]);  getitem_1129 = None
        empty_665 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_75 = torch.ops.aten.expand.default(empty_665, [512, 512, 28, 28]);  empty_665 = None
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(view_2802, expand_75, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_75 = convert_element_type_16 = None
        getitem_1131 = convolution_backward_74[0];  convolution_backward_74 = None
        empty_666 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_76 = torch.ops.aten.expand.default(empty_666, [128, 512, 1, 1]);  empty_666 = None
        view_2803 = torch.ops.aten.view.default(getitem_1130, [512, 784, 512]);  getitem_1130 = None
        view_2804 = torch.ops.aten.view.default(view_2803, [512, 512, 28, 28]);  view_2803 = None
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(view_2802, view_2804, expand_76, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2802 = view_2804 = expand_76 = None
        getitem_1135 = convolution_backward_75[1];  convolution_backward_75 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(getitem_1135, torch.float32);  getitem_1135 = None
        add_239 = torch.ops.aten.add.Tensor(mul_404, getitem_1131);  mul_404 = getitem_1131 = None
        triton_kernel_wrapper_functional_proxy_500 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 811, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_190, 'Y_ptr': full_default_395, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_190 = None
        getitem_1137 = triton_kernel_wrapper_functional_proxy_500['Y_ptr'];  triton_kernel_wrapper_functional_proxy_500 = None
        view_2807 = torch.ops.aten.view.default(getitem_1137, [512, 784, 512]);  getitem_1137 = None
        view_2808 = torch.ops.aten.view.default(view_2807, [512, 512, 28, 28]);  view_2807 = None
        mul_407 = torch.ops.aten.mul.Tensor(add_239, view_2808);  add_239 = view_2808 = None
        empty_667 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_501 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 588, constant_args_idx = 812, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_185, 'S_ptr': getitem_186, 'M_ptr': getitem_187, 'Y_ptr': empty_667, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_185 = getitem_186 = getitem_187 = empty_667 = None
        getitem_1138 = triton_kernel_wrapper_functional_proxy_501['Y_ptr'];  triton_kernel_wrapper_functional_proxy_501 = None
        view_2823 = torch.ops.aten.view.default(mul_407, [512, 512, 784]);  mul_407 = None
        view_2824 = torch.ops.aten.view.default(getitem_1138, [512, 784, 512]);  getitem_1138 = None
        view_2825 = torch.ops.aten.view.default(view_2824, [512, 512, 784]);  view_2824 = None
        triton_kernel_wrapper_functional_proxy_502 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 589, constant_args_idx = 813, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2825, 'DY': view_2823, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1139 = triton_kernel_wrapper_functional_proxy_502['DBETA']
        getitem_1140 = triton_kernel_wrapper_functional_proxy_502['DGAMMA'];  triton_kernel_wrapper_functional_proxy_502 = None
        empty_668 = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_195 = torch.ops.aten.permute.default(empty_668, [0, 1, 2]);  empty_668 = None
        triton_kernel_wrapper_functional_proxy_503 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 590, constant_args_idx = 814, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2825, 'DY': view_2823, 'INVSTD': rsqrt_14, 'GAMMA': primals_88, 'DBETA': getitem_1139, 'DGAMMA': getitem_1140, 'DX': permute_195, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2825 = rsqrt_14 = primals_88 = permute_195 = None
        getitem_1141 = triton_kernel_wrapper_functional_proxy_503['DX'];  triton_kernel_wrapper_functional_proxy_503 = None
        convert_element_type_default_29 = torch.ops.prims.convert_element_type.default(getitem_1140, torch.float32);  getitem_1140 = None
        convert_element_type_default_28 = torch.ops.prims.convert_element_type.default(getitem_1139, torch.float32);  getitem_1139 = None
        empty_669 = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_504 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 591, constant_args_idx = 815, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_178, 'S_ptr': getitem_179, 'M_ptr': getitem_180, 'Y_ptr': empty_669, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_178 = getitem_179 = getitem_180 = empty_669 = None
        getitem_1142 = triton_kernel_wrapper_functional_proxy_504['Y_ptr'];  triton_kernel_wrapper_functional_proxy_504 = None
        view_2841 = torch.ops.aten.view.default(getitem_1141, [512, 512, 28, 28]);  getitem_1141 = None
        empty_670 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_77 = torch.ops.aten.expand.default(empty_670, [512, 256, 56, 56]);  empty_670 = None
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(view_2841, expand_77, convert_element_type_15, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_77 = convert_element_type_15 = None
        getitem_1143 = convolution_backward_76[0];  convolution_backward_76 = None
        empty_671 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_78 = torch.ops.aten.expand.default(empty_671, [512, 256, 1, 1]);  empty_671 = None
        view_2842 = torch.ops.aten.view.default(getitem_1142, [512, 1568, 512]);  getitem_1142 = None
        view_2843 = torch.ops.aten.view.default(view_2842, [512, 256, 56, 56]);  view_2842 = None
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(view_2841, view_2843, expand_78, None, [2, 2], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2841 = view_2843 = expand_78 = None
        getitem_1147 = convolution_backward_77[1];  convolution_backward_77 = None
        convert_element_type_257 = torch.ops.prims.convert_element_type.default(getitem_1147, torch.float32);  getitem_1147 = None
        empty_672 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_505 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 592, constant_args_idx = 816, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_175, 'S_ptr': getitem_176, 'M_ptr': getitem_177, 'Y_ptr': empty_672, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_175 = getitem_176 = getitem_177 = empty_672 = None
        getitem_1149 = triton_kernel_wrapper_functional_proxy_505['Y_ptr'];  triton_kernel_wrapper_functional_proxy_505 = None
        view_2859 = torch.ops.aten.view.default(getitem_1149, [512, 784, 512]);  getitem_1149 = None
        view_2860 = torch.ops.aten.view.default(view_2859, [512, 512, 784]);  view_2859 = None
        triton_kernel_wrapper_functional_proxy_506 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 593, constant_args_idx = 817, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2860, 'DY': view_2823, 'DBETA': full_default_76, 'DGAMMA': full_default_76, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_76 = None
        getitem_1150 = triton_kernel_wrapper_functional_proxy_506['DBETA']
        getitem_1151 = triton_kernel_wrapper_functional_proxy_506['DGAMMA'];  triton_kernel_wrapper_functional_proxy_506 = None
        empty_673 = torch.ops.aten.empty.memory_format([512, 512, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_196 = torch.ops.aten.permute.default(empty_673, [0, 1, 2]);  empty_673 = None
        triton_kernel_wrapper_functional_proxy_507 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 594, constant_args_idx = 818, grid = [(512, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2860, 'DY': view_2823, 'INVSTD': rsqrt_13, 'GAMMA': primals_82, 'DBETA': getitem_1150, 'DGAMMA': getitem_1151, 'DX': permute_196, 'M': 401408, 'HW': 784, 'stride_n': 401408, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2860 = view_2823 = rsqrt_13 = primals_82 = permute_196 = None
        getitem_1152 = triton_kernel_wrapper_functional_proxy_507['DX'];  triton_kernel_wrapper_functional_proxy_507 = None
        convert_element_type_default_27 = torch.ops.prims.convert_element_type.default(getitem_1151, torch.float32);  getitem_1151 = None
        convert_element_type_default_26 = torch.ops.prims.convert_element_type.default(getitem_1150, torch.float32);  getitem_1150 = None
        empty_674 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_508 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 595, constant_args_idx = 819, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_168, 'S_ptr': getitem_169, 'M_ptr': getitem_170, 'Y_ptr': empty_674, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_168 = getitem_169 = getitem_170 = empty_674 = None
        getitem_1153 = triton_kernel_wrapper_functional_proxy_508['Y_ptr'];  triton_kernel_wrapper_functional_proxy_508 = None
        view_2876 = torch.ops.aten.view.default(getitem_1152, [512, 512, 28, 28]);  getitem_1152 = None
        empty_675 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_79 = torch.ops.aten.expand.default(empty_675, [512, 128, 28, 28]);  empty_675 = None
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(view_2876, expand_79, convert_element_type_14, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_79 = convert_element_type_14 = None
        getitem_1154 = convolution_backward_78[0];  convolution_backward_78 = None
        empty_676 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_80 = torch.ops.aten.expand.default(empty_676, [512, 128, 1, 1]);  empty_676 = None
        view_2877 = torch.ops.aten.view.default(getitem_1153, [512, 196, 512]);  getitem_1153 = None
        view_2878 = torch.ops.aten.view.default(view_2877, [512, 128, 28, 28]);  view_2877 = None
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(view_2876, view_2878, expand_80, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2876 = view_2878 = expand_80 = None
        getitem_1158 = convolution_backward_79[1];  convolution_backward_79 = None
        convert_element_type_262 = torch.ops.prims.convert_element_type.default(getitem_1158, torch.float32);  getitem_1158 = None
        triton_kernel_wrapper_functional_proxy_509 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 820, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_167, 'Y_ptr': full_default_310, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_167 = full_default_310 = None
        getitem_1160 = triton_kernel_wrapper_functional_proxy_509['Y_ptr'];  triton_kernel_wrapper_functional_proxy_509 = None
        view_2881 = torch.ops.aten.view.default(getitem_1160, [512, 196, 512]);  getitem_1160 = None
        view_2882 = torch.ops.aten.view.default(view_2881, [512, 128, 28, 28]);  view_2881 = None
        mul_408 = torch.ops.aten.mul.Tensor(getitem_1154, view_2882);  getitem_1154 = view_2882 = None
        empty_677 = torch.ops.aten.empty.memory_format([100352, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_510 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 596, constant_args_idx = 821, grid = [(100352, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_162, 'S_ptr': getitem_163, 'M_ptr': getitem_164, 'Y_ptr': empty_677, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_162 = getitem_163 = getitem_164 = empty_677 = None
        getitem_1161 = triton_kernel_wrapper_functional_proxy_510['Y_ptr'];  triton_kernel_wrapper_functional_proxy_510 = None
        view_2897 = torch.ops.aten.view.default(mul_408, [512, 128, 784]);  mul_408 = None
        view_2898 = torch.ops.aten.view.default(getitem_1161, [512, 196, 512]);  getitem_1161 = None
        view_2899 = torch.ops.aten.view.default(view_2898, [512, 128, 784]);  view_2898 = None
        triton_kernel_wrapper_functional_proxy_511 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 597, constant_args_idx = 822, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2899, 'DY': view_2897, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1162 = triton_kernel_wrapper_functional_proxy_511['DBETA']
        getitem_1163 = triton_kernel_wrapper_functional_proxy_511['DGAMMA'];  triton_kernel_wrapper_functional_proxy_511 = None
        empty_678 = torch.ops.aten.empty.memory_format([512, 128, 784], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_197 = torch.ops.aten.permute.default(empty_678, [0, 1, 2]);  empty_678 = None
        triton_kernel_wrapper_functional_proxy_512 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 598, constant_args_idx = 823, grid = [(128, 392, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2899, 'DY': view_2897, 'INVSTD': rsqrt_12, 'GAMMA': primals_76, 'DBETA': getitem_1162, 'DGAMMA': getitem_1163, 'DX': permute_197, 'M': 401408, 'HW': 784, 'stride_n': 100352, 'stride_c': 784, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2899 = view_2897 = rsqrt_12 = primals_76 = permute_197 = None
        getitem_1164 = triton_kernel_wrapper_functional_proxy_512['DX'];  triton_kernel_wrapper_functional_proxy_512 = None
        convert_element_type_default_25 = torch.ops.prims.convert_element_type.default(getitem_1163, torch.float32);  getitem_1163 = None
        convert_element_type_default_24 = torch.ops.prims.convert_element_type.default(getitem_1162, torch.float32);  getitem_1162 = None
        empty_679 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_513 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 599, constant_args_idx = 824, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_155, 'S_ptr': getitem_156, 'M_ptr': getitem_157, 'Y_ptr': empty_679, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_155 = getitem_156 = getitem_157 = empty_679 = None
        getitem_1165 = triton_kernel_wrapper_functional_proxy_513['Y_ptr'];  triton_kernel_wrapper_functional_proxy_513 = None
        view_2915 = torch.ops.aten.view.default(getitem_1164, [512, 128, 28, 28]);  getitem_1164 = None
        empty_680 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_81 = torch.ops.aten.expand.default(empty_680, [512, 128, 56, 56]);  empty_680 = None
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(view_2915, expand_81, convert_element_type_13, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_81 = convert_element_type_13 = None
        getitem_1166 = convolution_backward_80[0];  convolution_backward_80 = None
        empty_681 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_82 = torch.ops.aten.expand.default(empty_681, [128, 128, 3, 3]);  empty_681 = None
        view_2916 = torch.ops.aten.view.default(getitem_1165, [512, 784, 512]);  getitem_1165 = None
        view_2917 = torch.ops.aten.view.default(view_2916, [512, 128, 56, 56]);  view_2916 = None
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(view_2915, view_2917, expand_82, None, [2, 2], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_2915 = view_2917 = expand_82 = None
        getitem_1170 = convolution_backward_81[1];  convolution_backward_81 = None
        convert_element_type_267 = torch.ops.prims.convert_element_type.default(getitem_1170, torch.float32);  getitem_1170 = None
        triton_kernel_wrapper_functional_proxy_514 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 825, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_154, 'Y_ptr': full_default_395, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_154 = full_default_395 = None
        getitem_1172 = triton_kernel_wrapper_functional_proxy_514['Y_ptr'];  triton_kernel_wrapper_functional_proxy_514 = None
        view_2920 = torch.ops.aten.view.default(getitem_1172, [512, 784, 512]);  getitem_1172 = None
        view_2921 = torch.ops.aten.view.default(view_2920, [512, 128, 56, 56]);  view_2920 = None
        mul_409 = torch.ops.aten.mul.Tensor(getitem_1166, view_2921);  getitem_1166 = view_2921 = None
        empty_682 = torch.ops.aten.empty.memory_format([401408, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_515 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 600, constant_args_idx = 826, grid = [(401408, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_149, 'S_ptr': getitem_150, 'M_ptr': getitem_151, 'Y_ptr': empty_682, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_149 = getitem_150 = getitem_151 = empty_682 = None
        getitem_1173 = triton_kernel_wrapper_functional_proxy_515['Y_ptr'];  triton_kernel_wrapper_functional_proxy_515 = None
        view_2936 = torch.ops.aten.view.default(mul_409, [512, 128, 3136]);  mul_409 = None
        view_2937 = torch.ops.aten.view.default(getitem_1173, [512, 784, 512]);  getitem_1173 = None
        view_2938 = torch.ops.aten.view.default(view_2937, [512, 128, 3136]);  view_2937 = None
        triton_kernel_wrapper_functional_proxy_516 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 601, constant_args_idx = 827, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2938, 'DY': view_2936, 'DBETA': full_default_64, 'DGAMMA': full_default_64, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_64 = None
        getitem_1174 = triton_kernel_wrapper_functional_proxy_516['DBETA']
        getitem_1175 = triton_kernel_wrapper_functional_proxy_516['DGAMMA'];  triton_kernel_wrapper_functional_proxy_516 = None
        empty_683 = torch.ops.aten.empty.memory_format([512, 128, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_198 = torch.ops.aten.permute.default(empty_683, [0, 1, 2]);  empty_683 = None
        triton_kernel_wrapper_functional_proxy_517 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 602, constant_args_idx = 828, grid = [(128, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2938, 'DY': view_2936, 'INVSTD': rsqrt_11, 'GAMMA': primals_70, 'DBETA': getitem_1174, 'DGAMMA': getitem_1175, 'DX': permute_198, 'M': 1605632, 'HW': 3136, 'stride_n': 401408, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2938 = view_2936 = rsqrt_11 = primals_70 = permute_198 = None
        getitem_1176 = triton_kernel_wrapper_functional_proxy_517['DX'];  triton_kernel_wrapper_functional_proxy_517 = None
        convert_element_type_default_23 = torch.ops.prims.convert_element_type.default(getitem_1175, torch.float32);  getitem_1175 = None
        convert_element_type_default_22 = torch.ops.prims.convert_element_type.default(getitem_1174, torch.float32);  getitem_1174 = None
        empty_684 = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_518 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 603, constant_args_idx = 829, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_142, 'S_ptr': getitem_143, 'M_ptr': getitem_144, 'Y_ptr': empty_684, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_142 = getitem_143 = getitem_144 = empty_684 = None
        getitem_1177 = triton_kernel_wrapper_functional_proxy_518['Y_ptr'];  triton_kernel_wrapper_functional_proxy_518 = None
        view_2954 = torch.ops.aten.view.default(getitem_1176, [512, 128, 56, 56]);  getitem_1176 = None
        empty_685 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_83 = torch.ops.aten.expand.default(empty_685, [512, 256, 56, 56]);  empty_685 = None
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(view_2954, expand_83, convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_83 = convert_element_type_12 = None
        getitem_1178 = convolution_backward_82[0];  convolution_backward_82 = None
        empty_686 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_84 = torch.ops.aten.expand.default(empty_686, [128, 256, 1, 1]);  empty_686 = None
        view_2955 = torch.ops.aten.view.default(getitem_1177, [512, 1568, 512]);  getitem_1177 = None
        view_2956 = torch.ops.aten.view.default(view_2955, [512, 256, 56, 56]);  view_2955 = None
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(view_2954, view_2956, expand_84, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2954 = view_2956 = expand_84 = None
        getitem_1182 = convolution_backward_83[1];  convolution_backward_83 = None
        convert_element_type_272 = torch.ops.prims.convert_element_type.default(getitem_1182, torch.float32);  getitem_1182 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_1143, getitem_1178);  getitem_1143 = getitem_1178 = None
        full_default_433 = torch.ops.aten.full.default([802816, 512], 0, dtype = torch.int8, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_519 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 830, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_141, 'Y_ptr': full_default_433, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_141 = None
        getitem_1184 = triton_kernel_wrapper_functional_proxy_519['Y_ptr'];  triton_kernel_wrapper_functional_proxy_519 = None
        view_2959 = torch.ops.aten.view.default(getitem_1184, [512, 1568, 512]);  getitem_1184 = None
        view_2960 = torch.ops.aten.view.default(view_2959, [512, 256, 56, 56]);  view_2959 = None
        mul_410 = torch.ops.aten.mul.Tensor(add_240, view_2960);  add_240 = view_2960 = None
        empty_687 = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_520 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 604, constant_args_idx = 831, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_136, 'S_ptr': getitem_137, 'M_ptr': getitem_138, 'Y_ptr': empty_687, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_136 = getitem_137 = getitem_138 = empty_687 = None
        getitem_1185 = triton_kernel_wrapper_functional_proxy_520['Y_ptr'];  triton_kernel_wrapper_functional_proxy_520 = None
        view_2975 = torch.ops.aten.view.default(mul_410, [512, 256, 3136])
        view_2976 = torch.ops.aten.view.default(getitem_1185, [512, 1568, 512]);  getitem_1185 = None
        view_2977 = torch.ops.aten.view.default(view_2976, [512, 256, 3136]);  view_2976 = None
        triton_kernel_wrapper_functional_proxy_521 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 605, constant_args_idx = 832, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2977, 'DY': view_2975, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1186 = triton_kernel_wrapper_functional_proxy_521['DBETA']
        getitem_1187 = triton_kernel_wrapper_functional_proxy_521['DGAMMA'];  triton_kernel_wrapper_functional_proxy_521 = None
        empty_688 = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_199 = torch.ops.aten.permute.default(empty_688, [0, 1, 2]);  empty_688 = None
        triton_kernel_wrapper_functional_proxy_522 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 606, constant_args_idx = 833, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_2977, 'DY': view_2975, 'INVSTD': rsqrt_10, 'GAMMA': primals_64, 'DBETA': getitem_1186, 'DGAMMA': getitem_1187, 'DX': permute_199, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_2977 = view_2975 = rsqrt_10 = primals_64 = permute_199 = None
        getitem_1188 = triton_kernel_wrapper_functional_proxy_522['DX'];  triton_kernel_wrapper_functional_proxy_522 = None
        convert_element_type_default_21 = torch.ops.prims.convert_element_type.default(getitem_1187, torch.float32);  getitem_1187 = None
        convert_element_type_default_20 = torch.ops.prims.convert_element_type.default(getitem_1186, torch.float32);  getitem_1186 = None
        empty_689 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_523 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 607, constant_args_idx = 834, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_129, 'S_ptr': getitem_130, 'M_ptr': getitem_131, 'Y_ptr': empty_689, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_129 = getitem_130 = getitem_131 = empty_689 = None
        getitem_1189 = triton_kernel_wrapper_functional_proxy_523['Y_ptr'];  triton_kernel_wrapper_functional_proxy_523 = None
        view_2993 = torch.ops.aten.view.default(getitem_1188, [512, 256, 56, 56]);  getitem_1188 = None
        empty_690 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_85 = torch.ops.aten.expand.default(empty_690, [512, 64, 56, 56]);  empty_690 = None
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(view_2993, expand_85, convert_element_type_11, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_85 = convert_element_type_11 = None
        getitem_1190 = convolution_backward_84[0];  convolution_backward_84 = None
        empty_691 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_86 = torch.ops.aten.expand.default(empty_691, [256, 64, 1, 1]);  empty_691 = None
        view_2994 = torch.ops.aten.view.default(getitem_1189, [512, 392, 512]);  getitem_1189 = None
        view_2995 = torch.ops.aten.view.default(view_2994, [512, 64, 56, 56]);  view_2994 = None
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(view_2993, view_2995, expand_86, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_2993 = view_2995 = expand_86 = None
        getitem_1194 = convolution_backward_85[1];  convolution_backward_85 = None
        convert_element_type_277 = torch.ops.prims.convert_element_type.default(getitem_1194, torch.float32);  getitem_1194 = None
        triton_kernel_wrapper_functional_proxy_524 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 835, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_128, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_128 = None
        getitem_1196 = triton_kernel_wrapper_functional_proxy_524['Y_ptr'];  triton_kernel_wrapper_functional_proxy_524 = None
        view_2998 = torch.ops.aten.view.default(getitem_1196, [512, 392, 512]);  getitem_1196 = None
        view_2999 = torch.ops.aten.view.default(view_2998, [512, 64, 56, 56]);  view_2998 = None
        mul_411 = torch.ops.aten.mul.Tensor(getitem_1190, view_2999);  getitem_1190 = view_2999 = None
        empty_692 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_525 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 608, constant_args_idx = 836, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_123, 'S_ptr': getitem_124, 'M_ptr': getitem_125, 'Y_ptr': empty_692, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_123 = getitem_124 = getitem_125 = empty_692 = None
        getitem_1197 = triton_kernel_wrapper_functional_proxy_525['Y_ptr'];  triton_kernel_wrapper_functional_proxy_525 = None
        view_3014 = torch.ops.aten.view.default(mul_411, [512, 64, 3136]);  mul_411 = None
        view_3015 = torch.ops.aten.view.default(getitem_1197, [512, 392, 512]);  getitem_1197 = None
        view_3016 = torch.ops.aten.view.default(view_3015, [512, 64, 3136]);  view_3015 = None
        full_default = torch.ops.aten.full.default([64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_526 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 609, constant_args_idx = 837, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3016, 'DY': view_3014, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1198 = triton_kernel_wrapper_functional_proxy_526['DBETA']
        getitem_1199 = triton_kernel_wrapper_functional_proxy_526['DGAMMA'];  triton_kernel_wrapper_functional_proxy_526 = None
        empty_693 = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_200 = torch.ops.aten.permute.default(empty_693, [0, 1, 2]);  empty_693 = None
        triton_kernel_wrapper_functional_proxy_527 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 610, constant_args_idx = 838, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3016, 'DY': view_3014, 'INVSTD': rsqrt_9, 'GAMMA': primals_58, 'DBETA': getitem_1198, 'DGAMMA': getitem_1199, 'DX': permute_200, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3016 = view_3014 = rsqrt_9 = primals_58 = permute_200 = None
        getitem_1200 = triton_kernel_wrapper_functional_proxy_527['DX'];  triton_kernel_wrapper_functional_proxy_527 = None
        convert_element_type_default_19 = torch.ops.prims.convert_element_type.default(getitem_1199, torch.float32);  getitem_1199 = None
        convert_element_type_default_18 = torch.ops.prims.convert_element_type.default(getitem_1198, torch.float32);  getitem_1198 = None
        empty_694 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_528 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 611, constant_args_idx = 839, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_116, 'S_ptr': getitem_117, 'M_ptr': getitem_118, 'Y_ptr': empty_694, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_116 = getitem_117 = getitem_118 = empty_694 = None
        getitem_1201 = triton_kernel_wrapper_functional_proxy_528['Y_ptr'];  triton_kernel_wrapper_functional_proxy_528 = None
        view_3032 = torch.ops.aten.view.default(getitem_1200, [512, 64, 56, 56]);  getitem_1200 = None
        empty_695 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_87 = torch.ops.aten.expand.default(empty_695, [512, 64, 56, 56]);  empty_695 = None
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(view_3032, expand_87, convert_element_type_10, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_87 = convert_element_type_10 = None
        getitem_1202 = convolution_backward_86[0];  convolution_backward_86 = None
        empty_696 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_88 = torch.ops.aten.expand.default(empty_696, [64, 64, 3, 3]);  empty_696 = None
        view_3033 = torch.ops.aten.view.default(getitem_1201, [512, 392, 512]);  getitem_1201 = None
        view_3034 = torch.ops.aten.view.default(view_3033, [512, 64, 56, 56]);  view_3033 = None
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(view_3032, view_3034, expand_88, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3032 = view_3034 = expand_88 = None
        getitem_1206 = convolution_backward_87[1];  convolution_backward_87 = None
        convert_element_type_282 = torch.ops.prims.convert_element_type.default(getitem_1206, torch.float32);  getitem_1206 = None
        triton_kernel_wrapper_functional_proxy_529 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 840, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_115, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_115 = None
        getitem_1208 = triton_kernel_wrapper_functional_proxy_529['Y_ptr'];  triton_kernel_wrapper_functional_proxy_529 = None
        view_3037 = torch.ops.aten.view.default(getitem_1208, [512, 392, 512]);  getitem_1208 = None
        view_3038 = torch.ops.aten.view.default(view_3037, [512, 64, 56, 56]);  view_3037 = None
        mul_412 = torch.ops.aten.mul.Tensor(getitem_1202, view_3038);  getitem_1202 = view_3038 = None
        empty_697 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_530 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 612, constant_args_idx = 841, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_110, 'S_ptr': getitem_111, 'M_ptr': getitem_112, 'Y_ptr': empty_697, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_110 = getitem_111 = getitem_112 = empty_697 = None
        getitem_1209 = triton_kernel_wrapper_functional_proxy_530['Y_ptr'];  triton_kernel_wrapper_functional_proxy_530 = None
        view_3053 = torch.ops.aten.view.default(mul_412, [512, 64, 3136]);  mul_412 = None
        view_3054 = torch.ops.aten.view.default(getitem_1209, [512, 392, 512]);  getitem_1209 = None
        view_3055 = torch.ops.aten.view.default(view_3054, [512, 64, 3136]);  view_3054 = None
        triton_kernel_wrapper_functional_proxy_531 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 613, constant_args_idx = 842, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3055, 'DY': view_3053, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1210 = triton_kernel_wrapper_functional_proxy_531['DBETA']
        getitem_1211 = triton_kernel_wrapper_functional_proxy_531['DGAMMA'];  triton_kernel_wrapper_functional_proxy_531 = None
        empty_698 = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_201 = torch.ops.aten.permute.default(empty_698, [0, 1, 2]);  empty_698 = None
        triton_kernel_wrapper_functional_proxy_532 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 614, constant_args_idx = 843, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3055, 'DY': view_3053, 'INVSTD': rsqrt_8, 'GAMMA': primals_52, 'DBETA': getitem_1210, 'DGAMMA': getitem_1211, 'DX': permute_201, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3055 = view_3053 = rsqrt_8 = primals_52 = permute_201 = None
        getitem_1212 = triton_kernel_wrapper_functional_proxy_532['DX'];  triton_kernel_wrapper_functional_proxy_532 = None
        convert_element_type_default_17 = torch.ops.prims.convert_element_type.default(getitem_1211, torch.float32);  getitem_1211 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(getitem_1210, torch.float32);  getitem_1210 = None
        empty_699 = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_533 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 615, constant_args_idx = 844, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_103, 'S_ptr': getitem_104, 'M_ptr': getitem_105, 'Y_ptr': empty_699, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_103 = getitem_104 = getitem_105 = empty_699 = None
        getitem_1213 = triton_kernel_wrapper_functional_proxy_533['Y_ptr'];  triton_kernel_wrapper_functional_proxy_533 = None
        view_3071 = torch.ops.aten.view.default(getitem_1212, [512, 64, 56, 56]);  getitem_1212 = None
        empty_700 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_89 = torch.ops.aten.expand.default(empty_700, [512, 256, 56, 56]);  empty_700 = None
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(view_3071, expand_89, convert_element_type_9, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_89 = convert_element_type_9 = None
        getitem_1214 = convolution_backward_88[0];  convolution_backward_88 = None
        empty_701 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_90 = torch.ops.aten.expand.default(empty_701, [64, 256, 1, 1]);  empty_701 = None
        view_3072 = torch.ops.aten.view.default(getitem_1213, [512, 1568, 512]);  getitem_1213 = None
        view_3073 = torch.ops.aten.view.default(view_3072, [512, 256, 56, 56]);  view_3072 = None
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(view_3071, view_3073, expand_90, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3071 = view_3073 = expand_90 = None
        getitem_1218 = convolution_backward_89[1];  convolution_backward_89 = None
        convert_element_type_287 = torch.ops.prims.convert_element_type.default(getitem_1218, torch.float32);  getitem_1218 = None
        add_241 = torch.ops.aten.add.Tensor(mul_410, getitem_1214);  mul_410 = getitem_1214 = None
        triton_kernel_wrapper_functional_proxy_534 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 845, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_102, 'Y_ptr': full_default_433, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_102 = None
        getitem_1220 = triton_kernel_wrapper_functional_proxy_534['Y_ptr'];  triton_kernel_wrapper_functional_proxy_534 = None
        view_3076 = torch.ops.aten.view.default(getitem_1220, [512, 1568, 512]);  getitem_1220 = None
        view_3077 = torch.ops.aten.view.default(view_3076, [512, 256, 56, 56]);  view_3076 = None
        mul_413 = torch.ops.aten.mul.Tensor(add_241, view_3077);  add_241 = view_3077 = None
        empty_702 = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_535 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 616, constant_args_idx = 846, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_97, 'S_ptr': getitem_98, 'M_ptr': getitem_99, 'Y_ptr': empty_702, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_97 = getitem_98 = getitem_99 = empty_702 = None
        getitem_1221 = triton_kernel_wrapper_functional_proxy_535['Y_ptr'];  triton_kernel_wrapper_functional_proxy_535 = None
        view_3092 = torch.ops.aten.view.default(mul_413, [512, 256, 3136])
        view_3093 = torch.ops.aten.view.default(getitem_1221, [512, 1568, 512]);  getitem_1221 = None
        view_3094 = torch.ops.aten.view.default(view_3093, [512, 256, 3136]);  view_3093 = None
        triton_kernel_wrapper_functional_proxy_536 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 617, constant_args_idx = 847, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3094, 'DY': view_3092, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1222 = triton_kernel_wrapper_functional_proxy_536['DBETA']
        getitem_1223 = triton_kernel_wrapper_functional_proxy_536['DGAMMA'];  triton_kernel_wrapper_functional_proxy_536 = None
        empty_703 = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_202 = torch.ops.aten.permute.default(empty_703, [0, 1, 2]);  empty_703 = None
        triton_kernel_wrapper_functional_proxy_537 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 618, constant_args_idx = 848, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3094, 'DY': view_3092, 'INVSTD': rsqrt_7, 'GAMMA': primals_46, 'DBETA': getitem_1222, 'DGAMMA': getitem_1223, 'DX': permute_202, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3094 = view_3092 = rsqrt_7 = primals_46 = permute_202 = None
        getitem_1224 = triton_kernel_wrapper_functional_proxy_537['DX'];  triton_kernel_wrapper_functional_proxy_537 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(getitem_1223, torch.float32);  getitem_1223 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(getitem_1222, torch.float32);  getitem_1222 = None
        empty_704 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_538 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 619, constant_args_idx = 849, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_90, 'S_ptr': getitem_91, 'M_ptr': getitem_92, 'Y_ptr': empty_704, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_90 = getitem_91 = getitem_92 = empty_704 = None
        getitem_1225 = triton_kernel_wrapper_functional_proxy_538['Y_ptr'];  triton_kernel_wrapper_functional_proxy_538 = None
        view_3110 = torch.ops.aten.view.default(getitem_1224, [512, 256, 56, 56]);  getitem_1224 = None
        empty_705 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_91 = torch.ops.aten.expand.default(empty_705, [512, 64, 56, 56]);  empty_705 = None
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(view_3110, expand_91, convert_element_type_8, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_91 = convert_element_type_8 = None
        getitem_1226 = convolution_backward_90[0];  convolution_backward_90 = None
        empty_706 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_92 = torch.ops.aten.expand.default(empty_706, [256, 64, 1, 1]);  empty_706 = None
        view_3111 = torch.ops.aten.view.default(getitem_1225, [512, 392, 512]);  getitem_1225 = None
        view_3112 = torch.ops.aten.view.default(view_3111, [512, 64, 56, 56]);  view_3111 = None
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(view_3110, view_3112, expand_92, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3110 = view_3112 = expand_92 = None
        getitem_1230 = convolution_backward_91[1];  convolution_backward_91 = None
        convert_element_type_292 = torch.ops.prims.convert_element_type.default(getitem_1230, torch.float32);  getitem_1230 = None
        triton_kernel_wrapper_functional_proxy_539 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 850, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_89, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_89 = None
        getitem_1232 = triton_kernel_wrapper_functional_proxy_539['Y_ptr'];  triton_kernel_wrapper_functional_proxy_539 = None
        view_3115 = torch.ops.aten.view.default(getitem_1232, [512, 392, 512]);  getitem_1232 = None
        view_3116 = torch.ops.aten.view.default(view_3115, [512, 64, 56, 56]);  view_3115 = None
        mul_414 = torch.ops.aten.mul.Tensor(getitem_1226, view_3116);  getitem_1226 = view_3116 = None
        empty_707 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_540 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 620, constant_args_idx = 851, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_84, 'S_ptr': getitem_85, 'M_ptr': getitem_86, 'Y_ptr': empty_707, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_84 = getitem_85 = getitem_86 = empty_707 = None
        getitem_1233 = triton_kernel_wrapper_functional_proxy_540['Y_ptr'];  triton_kernel_wrapper_functional_proxy_540 = None
        view_3131 = torch.ops.aten.view.default(mul_414, [512, 64, 3136]);  mul_414 = None
        view_3132 = torch.ops.aten.view.default(getitem_1233, [512, 392, 512]);  getitem_1233 = None
        view_3133 = torch.ops.aten.view.default(view_3132, [512, 64, 3136]);  view_3132 = None
        triton_kernel_wrapper_functional_proxy_541 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 621, constant_args_idx = 852, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3133, 'DY': view_3131, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1234 = triton_kernel_wrapper_functional_proxy_541['DBETA']
        getitem_1235 = triton_kernel_wrapper_functional_proxy_541['DGAMMA'];  triton_kernel_wrapper_functional_proxy_541 = None
        empty_708 = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_203 = torch.ops.aten.permute.default(empty_708, [0, 1, 2]);  empty_708 = None
        triton_kernel_wrapper_functional_proxy_542 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 622, constant_args_idx = 853, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3133, 'DY': view_3131, 'INVSTD': rsqrt_6, 'GAMMA': primals_40, 'DBETA': getitem_1234, 'DGAMMA': getitem_1235, 'DX': permute_203, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3133 = view_3131 = rsqrt_6 = primals_40 = permute_203 = None
        getitem_1236 = triton_kernel_wrapper_functional_proxy_542['DX'];  triton_kernel_wrapper_functional_proxy_542 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(getitem_1235, torch.float32);  getitem_1235 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(getitem_1234, torch.float32);  getitem_1234 = None
        empty_709 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_543 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 623, constant_args_idx = 854, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_77, 'S_ptr': getitem_78, 'M_ptr': getitem_79, 'Y_ptr': empty_709, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_77 = getitem_78 = getitem_79 = empty_709 = None
        getitem_1237 = triton_kernel_wrapper_functional_proxy_543['Y_ptr'];  triton_kernel_wrapper_functional_proxy_543 = None
        view_3149 = torch.ops.aten.view.default(getitem_1236, [512, 64, 56, 56]);  getitem_1236 = None
        empty_710 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_93 = torch.ops.aten.expand.default(empty_710, [512, 64, 56, 56]);  empty_710 = None
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(view_3149, expand_93, convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_93 = convert_element_type_7 = None
        getitem_1238 = convolution_backward_92[0];  convolution_backward_92 = None
        empty_711 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_94 = torch.ops.aten.expand.default(empty_711, [64, 64, 3, 3]);  empty_711 = None
        view_3150 = torch.ops.aten.view.default(getitem_1237, [512, 392, 512]);  getitem_1237 = None
        view_3151 = torch.ops.aten.view.default(view_3150, [512, 64, 56, 56]);  view_3150 = None
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(view_3149, view_3151, expand_94, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3149 = view_3151 = expand_94 = None
        getitem_1242 = convolution_backward_93[1];  convolution_backward_93 = None
        convert_element_type_297 = torch.ops.prims.convert_element_type.default(getitem_1242, torch.float32);  getitem_1242 = None
        triton_kernel_wrapper_functional_proxy_544 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 855, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_76, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_76 = None
        getitem_1244 = triton_kernel_wrapper_functional_proxy_544['Y_ptr'];  triton_kernel_wrapper_functional_proxy_544 = None
        view_3154 = torch.ops.aten.view.default(getitem_1244, [512, 392, 512]);  getitem_1244 = None
        view_3155 = torch.ops.aten.view.default(view_3154, [512, 64, 56, 56]);  view_3154 = None
        mul_415 = torch.ops.aten.mul.Tensor(getitem_1238, view_3155);  getitem_1238 = view_3155 = None
        empty_712 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_545 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 624, constant_args_idx = 856, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_71, 'S_ptr': getitem_72, 'M_ptr': getitem_73, 'Y_ptr': empty_712, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_71 = getitem_72 = getitem_73 = empty_712 = None
        getitem_1245 = triton_kernel_wrapper_functional_proxy_545['Y_ptr'];  triton_kernel_wrapper_functional_proxy_545 = None
        view_3170 = torch.ops.aten.view.default(mul_415, [512, 64, 3136]);  mul_415 = None
        view_3171 = torch.ops.aten.view.default(getitem_1245, [512, 392, 512]);  getitem_1245 = None
        view_3172 = torch.ops.aten.view.default(view_3171, [512, 64, 3136]);  view_3171 = None
        triton_kernel_wrapper_functional_proxy_546 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 625, constant_args_idx = 857, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3172, 'DY': view_3170, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1246 = triton_kernel_wrapper_functional_proxy_546['DBETA']
        getitem_1247 = triton_kernel_wrapper_functional_proxy_546['DGAMMA'];  triton_kernel_wrapper_functional_proxy_546 = None
        empty_713 = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_204 = torch.ops.aten.permute.default(empty_713, [0, 1, 2]);  empty_713 = None
        triton_kernel_wrapper_functional_proxy_547 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 626, constant_args_idx = 858, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3172, 'DY': view_3170, 'INVSTD': rsqrt_5, 'GAMMA': primals_34, 'DBETA': getitem_1246, 'DGAMMA': getitem_1247, 'DX': permute_204, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3172 = view_3170 = rsqrt_5 = primals_34 = permute_204 = None
        getitem_1248 = triton_kernel_wrapper_functional_proxy_547['DX'];  triton_kernel_wrapper_functional_proxy_547 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(getitem_1247, torch.float32);  getitem_1247 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(getitem_1246, torch.float32);  getitem_1246 = None
        empty_714 = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_548 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 627, constant_args_idx = 859, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_64, 'S_ptr': getitem_65, 'M_ptr': getitem_66, 'Y_ptr': empty_714, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_64 = getitem_65 = getitem_66 = empty_714 = None
        getitem_1249 = triton_kernel_wrapper_functional_proxy_548['Y_ptr'];  triton_kernel_wrapper_functional_proxy_548 = None
        view_3188 = torch.ops.aten.view.default(getitem_1248, [512, 64, 56, 56]);  getitem_1248 = None
        empty_715 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_95 = torch.ops.aten.expand.default(empty_715, [512, 256, 56, 56]);  empty_715 = None
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(view_3188, expand_95, convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_95 = convert_element_type_6 = None
        getitem_1250 = convolution_backward_94[0];  convolution_backward_94 = None
        empty_716 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_96 = torch.ops.aten.expand.default(empty_716, [64, 256, 1, 1]);  empty_716 = None
        view_3189 = torch.ops.aten.view.default(getitem_1249, [512, 1568, 512]);  getitem_1249 = None
        view_3190 = torch.ops.aten.view.default(view_3189, [512, 256, 56, 56]);  view_3189 = None
        convolution_backward_95 = torch.ops.aten.convolution_backward.default(view_3188, view_3190, expand_96, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3188 = view_3190 = expand_96 = None
        getitem_1254 = convolution_backward_95[1];  convolution_backward_95 = None
        convert_element_type_302 = torch.ops.prims.convert_element_type.default(getitem_1254, torch.float32);  getitem_1254 = None
        add_242 = torch.ops.aten.add.Tensor(mul_413, getitem_1250);  mul_413 = getitem_1250 = None
        triton_kernel_wrapper_functional_proxy_549 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 860, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_63, 'Y_ptr': full_default_433, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_63 = None
        getitem_1256 = triton_kernel_wrapper_functional_proxy_549['Y_ptr'];  triton_kernel_wrapper_functional_proxy_549 = None
        view_3193 = torch.ops.aten.view.default(getitem_1256, [512, 1568, 512]);  getitem_1256 = None
        view_3194 = torch.ops.aten.view.default(view_3193, [512, 256, 56, 56]);  view_3193 = None
        mul_416 = torch.ops.aten.mul.Tensor(add_242, view_3194);  add_242 = view_3194 = None
        empty_717 = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_550 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 628, constant_args_idx = 861, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_58, 'S_ptr': getitem_59, 'M_ptr': getitem_60, 'Y_ptr': empty_717, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_58 = getitem_59 = getitem_60 = empty_717 = None
        getitem_1257 = triton_kernel_wrapper_functional_proxy_550['Y_ptr'];  triton_kernel_wrapper_functional_proxy_550 = None
        view_3209 = torch.ops.aten.view.default(mul_416, [512, 256, 3136]);  mul_416 = None
        view_3210 = torch.ops.aten.view.default(getitem_1257, [512, 1568, 512]);  getitem_1257 = None
        view_3211 = torch.ops.aten.view.default(view_3210, [512, 256, 3136]);  view_3210 = None
        triton_kernel_wrapper_functional_proxy_551 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 629, constant_args_idx = 862, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3211, 'DY': view_3209, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1258 = triton_kernel_wrapper_functional_proxy_551['DBETA']
        getitem_1259 = triton_kernel_wrapper_functional_proxy_551['DGAMMA'];  triton_kernel_wrapper_functional_proxy_551 = None
        empty_718 = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_205 = torch.ops.aten.permute.default(empty_718, [0, 1, 2]);  empty_718 = None
        triton_kernel_wrapper_functional_proxy_552 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 630, constant_args_idx = 863, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3211, 'DY': view_3209, 'INVSTD': rsqrt_4, 'GAMMA': primals_28, 'DBETA': getitem_1258, 'DGAMMA': getitem_1259, 'DX': permute_205, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3211 = rsqrt_4 = primals_28 = permute_205 = None
        getitem_1260 = triton_kernel_wrapper_functional_proxy_552['DX'];  triton_kernel_wrapper_functional_proxy_552 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(getitem_1259, torch.float32);  getitem_1259 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(getitem_1258, torch.float32);  getitem_1258 = None
        empty_719 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_553 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 631, constant_args_idx = 864, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_51, 'S_ptr': getitem_52, 'M_ptr': getitem_53, 'Y_ptr': empty_719, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_51 = getitem_52 = getitem_53 = empty_719 = None
        getitem_1261 = triton_kernel_wrapper_functional_proxy_553['Y_ptr'];  triton_kernel_wrapper_functional_proxy_553 = None
        view_3227 = torch.ops.aten.view.default(getitem_1260, [512, 256, 56, 56]);  getitem_1260 = None
        empty_720 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_97 = torch.ops.aten.expand.default(empty_720, [512, 64, 56, 56]);  empty_720 = None
        convolution_backward_96 = torch.ops.aten.convolution_backward.default(view_3227, expand_97, convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_97 = convert_element_type_5 = None
        getitem_1262 = convolution_backward_96[0];  convolution_backward_96 = None
        empty_721 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_98 = torch.ops.aten.expand.default(empty_721, [256, 64, 1, 1]);  empty_721 = None
        view_3228 = torch.ops.aten.view.default(getitem_1261, [512, 392, 512]);  getitem_1261 = None
        view_3229 = torch.ops.aten.view.default(view_3228, [512, 64, 56, 56]);  view_3228 = None
        convolution_backward_97 = torch.ops.aten.convolution_backward.default(view_3227, view_3229, expand_98, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3227 = view_3229 = expand_98 = None
        getitem_1266 = convolution_backward_97[1];  convolution_backward_97 = None
        convert_element_type_307 = torch.ops.prims.convert_element_type.default(getitem_1266, torch.float32);  getitem_1266 = None
        empty_722 = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_554 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 632, constant_args_idx = 865, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_48, 'S_ptr': getitem_49, 'M_ptr': getitem_50, 'Y_ptr': empty_722, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_48 = getitem_49 = getitem_50 = empty_722 = None
        getitem_1268 = triton_kernel_wrapper_functional_proxy_554['Y_ptr'];  triton_kernel_wrapper_functional_proxy_554 = None
        view_3245 = torch.ops.aten.view.default(getitem_1268, [512, 1568, 512]);  getitem_1268 = None
        view_3246 = torch.ops.aten.view.default(view_3245, [512, 256, 3136]);  view_3245 = None
        triton_kernel_wrapper_functional_proxy_555 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 633, constant_args_idx = 866, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3246, 'DY': view_3209, 'DBETA': full_default_18, 'DGAMMA': full_default_18, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default_18 = None
        getitem_1269 = triton_kernel_wrapper_functional_proxy_555['DBETA']
        getitem_1270 = triton_kernel_wrapper_functional_proxy_555['DGAMMA'];  triton_kernel_wrapper_functional_proxy_555 = None
        empty_723 = torch.ops.aten.empty.memory_format([512, 256, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_206 = torch.ops.aten.permute.default(empty_723, [0, 1, 2]);  empty_723 = None
        triton_kernel_wrapper_functional_proxy_556 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 634, constant_args_idx = 867, grid = [(256, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3246, 'DY': view_3209, 'INVSTD': rsqrt_3, 'GAMMA': primals_22, 'DBETA': getitem_1269, 'DGAMMA': getitem_1270, 'DX': permute_206, 'M': 1605632, 'HW': 3136, 'stride_n': 802816, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3246 = view_3209 = rsqrt_3 = primals_22 = permute_206 = None
        getitem_1271 = triton_kernel_wrapper_functional_proxy_556['DX'];  triton_kernel_wrapper_functional_proxy_556 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(getitem_1270, torch.float32);  getitem_1270 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(getitem_1269, torch.float32);  getitem_1269 = None
        empty_724 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_557 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 635, constant_args_idx = 868, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_41, 'S_ptr': getitem_42, 'M_ptr': getitem_43, 'Y_ptr': empty_724, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_41 = getitem_42 = getitem_43 = empty_724 = None
        getitem_1272 = triton_kernel_wrapper_functional_proxy_557['Y_ptr'];  triton_kernel_wrapper_functional_proxy_557 = None
        view_3262 = torch.ops.aten.view.default(getitem_1271, [512, 256, 56, 56]);  getitem_1271 = None
        empty_725 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_99 = torch.ops.aten.expand.default(empty_725, [512, 64, 56, 56]);  empty_725 = None
        convolution_backward_98 = torch.ops.aten.convolution_backward.default(view_3262, expand_99, convert_element_type_4, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_99 = convert_element_type_4 = None
        getitem_1273 = convolution_backward_98[0];  convolution_backward_98 = None
        empty_726 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_100 = torch.ops.aten.expand.default(empty_726, [256, 64, 1, 1]);  empty_726 = None
        view_3263 = torch.ops.aten.view.default(getitem_1272, [512, 392, 512]);  getitem_1272 = None
        view_3264 = torch.ops.aten.view.default(view_3263, [512, 64, 56, 56]);  view_3263 = None
        convolution_backward_99 = torch.ops.aten.convolution_backward.default(view_3262, view_3264, expand_100, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3262 = view_3264 = expand_100 = None
        getitem_1277 = convolution_backward_99[1];  convolution_backward_99 = None
        convert_element_type_312 = torch.ops.prims.convert_element_type.default(getitem_1277, torch.float32);  getitem_1277 = None
        triton_kernel_wrapper_functional_proxy_558 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 869, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_40, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_40 = None
        getitem_1279 = triton_kernel_wrapper_functional_proxy_558['Y_ptr'];  triton_kernel_wrapper_functional_proxy_558 = None
        view_3267 = torch.ops.aten.view.default(getitem_1279, [512, 392, 512]);  getitem_1279 = None
        view_3268 = torch.ops.aten.view.default(view_3267, [512, 64, 56, 56]);  view_3267 = None
        mul_417 = torch.ops.aten.mul.Tensor(getitem_1273, view_3268);  getitem_1273 = view_3268 = None
        empty_727 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_559 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 636, constant_args_idx = 870, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_35, 'S_ptr': getitem_36, 'M_ptr': getitem_37, 'Y_ptr': empty_727, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_35 = getitem_36 = getitem_37 = empty_727 = None
        getitem_1280 = triton_kernel_wrapper_functional_proxy_559['Y_ptr'];  triton_kernel_wrapper_functional_proxy_559 = None
        view_3283 = torch.ops.aten.view.default(mul_417, [512, 64, 3136]);  mul_417 = None
        view_3284 = torch.ops.aten.view.default(getitem_1280, [512, 392, 512]);  getitem_1280 = None
        view_3285 = torch.ops.aten.view.default(view_3284, [512, 64, 3136]);  view_3284 = None
        triton_kernel_wrapper_functional_proxy_560 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 637, constant_args_idx = 871, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3285, 'DY': view_3283, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1281 = triton_kernel_wrapper_functional_proxy_560['DBETA']
        getitem_1282 = triton_kernel_wrapper_functional_proxy_560['DGAMMA'];  triton_kernel_wrapper_functional_proxy_560 = None
        empty_728 = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_207 = torch.ops.aten.permute.default(empty_728, [0, 1, 2]);  empty_728 = None
        triton_kernel_wrapper_functional_proxy_561 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 638, constant_args_idx = 872, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3285, 'DY': view_3283, 'INVSTD': rsqrt_2, 'GAMMA': primals_16, 'DBETA': getitem_1281, 'DGAMMA': getitem_1282, 'DX': permute_207, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3285 = view_3283 = rsqrt_2 = primals_16 = permute_207 = None
        getitem_1283 = triton_kernel_wrapper_functional_proxy_561['DX'];  triton_kernel_wrapper_functional_proxy_561 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(getitem_1282, torch.float32);  getitem_1282 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(getitem_1281, torch.float32);  getitem_1281 = None
        empty_729 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_562 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 639, constant_args_idx = 873, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_28, 'S_ptr': getitem_29, 'M_ptr': getitem_30, 'Y_ptr': empty_729, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_28 = getitem_29 = getitem_30 = empty_729 = None
        getitem_1284 = triton_kernel_wrapper_functional_proxy_562['Y_ptr'];  triton_kernel_wrapper_functional_proxy_562 = None
        view_3301 = torch.ops.aten.view.default(getitem_1283, [512, 64, 56, 56]);  getitem_1283 = None
        empty_730 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_101 = torch.ops.aten.expand.default(empty_730, [512, 64, 56, 56]);  empty_730 = None
        convolution_backward_100 = torch.ops.aten.convolution_backward.default(view_3301, expand_101, convert_element_type_3, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [True, False, False]);  expand_101 = convert_element_type_3 = None
        getitem_1285 = convolution_backward_100[0];  convolution_backward_100 = None
        empty_731 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_102 = torch.ops.aten.expand.default(empty_731, [64, 64, 3, 3]);  empty_731 = None
        view_3302 = torch.ops.aten.view.default(getitem_1284, [512, 392, 512]);  getitem_1284 = None
        view_3303 = torch.ops.aten.view.default(view_3302, [512, 64, 56, 56]);  view_3302 = None
        convolution_backward_101 = torch.ops.aten.convolution_backward.default(view_3301, view_3303, expand_102, None, [1, 1], [1, 1], [1, 1], False, [0], 1, [False, True, False]);  view_3301 = view_3303 = expand_102 = None
        getitem_1289 = convolution_backward_101[1];  convolution_backward_101 = None
        convert_element_type_317 = torch.ops.prims.convert_element_type.default(getitem_1289, torch.float32);  getitem_1289 = None
        triton_kernel_wrapper_functional_proxy_563 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 874, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_27, 'Y_ptr': full_default_339, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_27 = full_default_339 = None
        getitem_1291 = triton_kernel_wrapper_functional_proxy_563['Y_ptr'];  triton_kernel_wrapper_functional_proxy_563 = None
        view_3306 = torch.ops.aten.view.default(getitem_1291, [512, 392, 512]);  getitem_1291 = None
        view_3307 = torch.ops.aten.view.default(view_3306, [512, 64, 56, 56]);  view_3306 = None
        mul_418 = torch.ops.aten.mul.Tensor(getitem_1285, view_3307);  getitem_1285 = view_3307 = None
        empty_732 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_564 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 640, constant_args_idx = 875, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_22, 'S_ptr': getitem_23, 'M_ptr': getitem_24, 'Y_ptr': empty_732, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_22 = getitem_23 = getitem_24 = empty_732 = None
        getitem_1292 = triton_kernel_wrapper_functional_proxy_564['Y_ptr'];  triton_kernel_wrapper_functional_proxy_564 = None
        view_3322 = torch.ops.aten.view.default(mul_418, [512, 64, 3136]);  mul_418 = None
        view_3323 = torch.ops.aten.view.default(getitem_1292, [512, 392, 512]);  getitem_1292 = None
        view_3324 = torch.ops.aten.view.default(view_3323, [512, 64, 3136]);  view_3323 = None
        triton_kernel_wrapper_functional_proxy_565 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 641, constant_args_idx = 876, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3324, 'DY': view_3322, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA'])
        getitem_1293 = triton_kernel_wrapper_functional_proxy_565['DBETA']
        getitem_1294 = triton_kernel_wrapper_functional_proxy_565['DGAMMA'];  triton_kernel_wrapper_functional_proxy_565 = None
        empty_733 = torch.ops.aten.empty.memory_format([512, 64, 3136], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_208 = torch.ops.aten.permute.default(empty_733, [0, 1, 2]);  empty_733 = None
        triton_kernel_wrapper_functional_proxy_566 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 642, constant_args_idx = 877, grid = [(64, 1568, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3324, 'DY': view_3322, 'INVSTD': rsqrt_1, 'GAMMA': primals_10, 'DBETA': getitem_1293, 'DGAMMA': getitem_1294, 'DX': permute_208, 'M': 1605632, 'HW': 3136, 'stride_n': 200704, 'stride_c': 3136, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3324 = view_3322 = rsqrt_1 = primals_10 = permute_208 = None
        getitem_1295 = triton_kernel_wrapper_functional_proxy_566['DX'];  triton_kernel_wrapper_functional_proxy_566 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(getitem_1294, torch.float32);  getitem_1294 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(getitem_1293, torch.float32);  getitem_1293 = None
        empty_734 = torch.ops.aten.empty.memory_format([200704, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_567 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 643, constant_args_idx = 878, grid = [(200704, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_15, 'S_ptr': getitem_16, 'M_ptr': getitem_17, 'Y_ptr': empty_734, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_15 = getitem_16 = getitem_17 = empty_734 = None
        getitem_1296 = triton_kernel_wrapper_functional_proxy_567['Y_ptr'];  triton_kernel_wrapper_functional_proxy_567 = None
        view_3340 = torch.ops.aten.view.default(getitem_1295, [512, 64, 56, 56]);  getitem_1295 = None
        empty_735 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_103 = torch.ops.aten.expand.default(empty_735, [512, 64, 56, 56]);  empty_735 = None
        convolution_backward_102 = torch.ops.aten.convolution_backward.default(view_3340, expand_103, convert_element_type_2, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [True, False, False]);  expand_103 = convert_element_type_2 = None
        getitem_1297 = convolution_backward_102[0];  convolution_backward_102 = None
        empty_736 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_104 = torch.ops.aten.expand.default(empty_736, [64, 64, 1, 1]);  empty_736 = None
        view_3341 = torch.ops.aten.view.default(getitem_1296, [512, 392, 512]);  getitem_1296 = None
        view_3342 = torch.ops.aten.view.default(view_3341, [512, 64, 56, 56]);  view_3341 = None
        convolution_backward_103 = torch.ops.aten.convolution_backward.default(view_3340, view_3342, expand_104, None, [1, 1], [0, 0], [1, 1], False, [0], 1, [False, True, False]);  view_3340 = view_3342 = expand_104 = None
        getitem_1301 = convolution_backward_103[1];  convolution_backward_103 = None
        convert_element_type_322 = torch.ops.prims.convert_element_type.default(getitem_1301, torch.float32);  getitem_1301 = None
        add_243 = torch.ops.aten.add.Tensor(getitem_1262, getitem_1297);  getitem_1262 = getitem_1297 = None
        _low_memory_max_pool_offsets_to_indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_14, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_14 = None
        max_pool2d_with_indices_backward = torch.ops.aten.max_pool2d_with_indices_backward.default(add_243, getitem_10, [3, 3], [2, 2], [1, 1], [1, 1], False, _low_memory_max_pool_offsets_to_indices);  add_243 = getitem_10 = _low_memory_max_pool_offsets_to_indices = None
        triton_kernel_wrapper_functional_proxy_568 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 433, constant_args_idx = 879, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_12, 'Y_ptr': full_default_433, 'stride_p0': 16, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 1, 'VPW': 32, 'NWORDS': 16}, tensors_to_clone = ['Y_ptr']);  getitem_12 = full_default_433 = None
        getitem_1303 = triton_kernel_wrapper_functional_proxy_568['Y_ptr'];  triton_kernel_wrapper_functional_proxy_568 = None
        view_3345 = torch.ops.aten.view.default(getitem_1303, [512, 1568, 512]);  getitem_1303 = None
        view_3346 = torch.ops.aten.view.default(view_3345, [512, 64, 112, 112]);  view_3345 = None
        mul_419 = torch.ops.aten.mul.Tensor(max_pool2d_with_indices_backward, view_3346);  max_pool2d_with_indices_backward = view_3346 = None
        empty_737 = torch.ops.aten.empty.memory_format([802816, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_569 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 644, constant_args_idx = 880, grid = [(802816, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem_7, 'S_ptr': getitem_8, 'M_ptr': getitem_9, 'Y_ptr': empty_737, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem_7 = getitem_8 = getitem_9 = empty_737 = None
        getitem_1304 = triton_kernel_wrapper_functional_proxy_569['Y_ptr'];  triton_kernel_wrapper_functional_proxy_569 = None
        view_3361 = torch.ops.aten.view.default(mul_419, [512, 64, 12544]);  mul_419 = None
        view_3362 = torch.ops.aten.view.default(getitem_1304, [512, 1568, 512]);  getitem_1304 = None
        view_3363 = torch.ops.aten.view.default(view_3362, [512, 64, 12544]);  view_3362 = None
        triton_kernel_wrapper_functional_proxy_570 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 645, constant_args_idx = 881, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3363, 'DY': view_3361, 'DBETA': full_default, 'DGAMMA': full_default, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024}, tensors_to_clone = ['DBETA', 'DGAMMA']);  full_default = None
        getitem_1305 = triton_kernel_wrapper_functional_proxy_570['DBETA']
        getitem_1306 = triton_kernel_wrapper_functional_proxy_570['DGAMMA'];  triton_kernel_wrapper_functional_proxy_570 = None
        empty_738 = torch.ops.aten.empty.memory_format([512, 64, 12544], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_209 = torch.ops.aten.permute.default(empty_738, [0, 1, 2]);  empty_738 = None
        triton_kernel_wrapper_functional_proxy_571 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 646, constant_args_idx = 882, grid = [(64, 6272, 1)], tma_descriptor_metadata = {}, kwargs = {'X_hat': view_3363, 'DY': view_3361, 'INVSTD': rsqrt, 'GAMMA': primals_4, 'DBETA': getitem_1305, 'DGAMMA': getitem_1306, 'DX': permute_209, 'M': 6422528, 'HW': 12544, 'stride_n': 802816, 'stride_c': 12544, 'BLOCK_M': 1024}, tensors_to_clone = ['DX']);  view_3363 = view_3361 = rsqrt = primals_4 = permute_209 = None
        getitem_1307 = triton_kernel_wrapper_functional_proxy_571['DX'];  triton_kernel_wrapper_functional_proxy_571 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(getitem_1306, torch.float32);  getitem_1306 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(getitem_1305, torch.float32);  getitem_1305 = None
        empty_739 = torch.ops.aten.empty.memory_format([150528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_572 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 647, constant_args_idx = 883, grid = [(150528, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'P_ptr': getitem, 'S_ptr': getitem_1, 'M_ptr': getitem_2, 'Y_ptr': empty_739, 'stride_p0': 32, 'stride_p1': 1, 'stride_y0': 512, 'stride_y1': 1, 'BITS': 2, 'VPW': 16, 'NWORDS': 32}, tensors_to_clone = ['Y_ptr']);  getitem = getitem_1 = getitem_2 = empty_739 = None
        getitem_1308 = triton_kernel_wrapper_functional_proxy_572['Y_ptr'];  triton_kernel_wrapper_functional_proxy_572 = None
        view_3379 = torch.ops.aten.view.default(getitem_1307, [512, 64, 112, 112]);  getitem_1307 = None
        empty_740 = torch.ops.aten.empty.memory_format([1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_105 = torch.ops.aten.expand.default(empty_740, [64, 3, 7, 7]);  empty_740 = None
        view_3380 = torch.ops.aten.view.default(getitem_1308, [512, 294, 512]);  getitem_1308 = None
        view_3381 = torch.ops.aten.view.default(view_3380, [512, 3, 224, 224]);  view_3380 = None
        convolution_backward_104 = torch.ops.aten.convolution_backward.default(view_3379, view_3381, expand_105, None, [2, 2], [3, 3], [1, 1], False, [0], 1, [False, True, False]);  view_3379 = view_3381 = expand_105 = None
        getitem_1310 = convolution_backward_104[1];  convolution_backward_104 = None
        convert_element_type_327 = torch.ops.prims.convert_element_type.default(getitem_1310, torch.float32);  getitem_1310 = None
        return (convert_element_type_327, None, None, convert_element_type_default_1, convert_element_type_default, None, None, convert_element_type_322, None, convert_element_type_default_3, convert_element_type_default_2, None, None, convert_element_type_317, None, convert_element_type_default_5, convert_element_type_default_4, None, None, convert_element_type_312, None, convert_element_type_default_7, convert_element_type_default_6, None, None, convert_element_type_307, None, convert_element_type_default_9, convert_element_type_default_8, None, None, convert_element_type_302, None, convert_element_type_default_11, convert_element_type_default_10, None, None, convert_element_type_297, None, convert_element_type_default_13, convert_element_type_default_12, None, None, convert_element_type_292, None, convert_element_type_default_15, convert_element_type_default_14, None, None, convert_element_type_287, None, convert_element_type_default_17, convert_element_type_default_16, None, None, convert_element_type_282, None, convert_element_type_default_19, convert_element_type_default_18, None, None, convert_element_type_277, None, convert_element_type_default_21, convert_element_type_default_20, None, None, convert_element_type_272, None, convert_element_type_default_23, convert_element_type_default_22, None, None, convert_element_type_267, None, convert_element_type_default_25, convert_element_type_default_24, None, None, convert_element_type_262, None, convert_element_type_default_27, convert_element_type_default_26, None, None, convert_element_type_257, None, convert_element_type_default_29, convert_element_type_default_28, None, None, convert_element_type_252, None, convert_element_type_default_31, convert_element_type_default_30, None, None, convert_element_type_247, None, convert_element_type_default_33, convert_element_type_default_32, None, None, convert_element_type_242, None, convert_element_type_default_35, convert_element_type_default_34, None, None, convert_element_type_237, None, convert_element_type_default_37, convert_element_type_default_36, None, None, convert_element_type_232, None, convert_element_type_default_39, convert_element_type_default_38, None, None, convert_element_type_227, None, convert_element_type_default_41, convert_element_type_default_40, None, None, convert_element_type_222, None, convert_element_type_default_43, convert_element_type_default_42, None, None, convert_element_type_217, None, convert_element_type_default_45, convert_element_type_default_44, None, None, convert_element_type_212, None, convert_element_type_default_47, convert_element_type_default_46, None, None, convert_element_type_207, None, convert_element_type_default_49, convert_element_type_default_48, None, None, convert_element_type_202, None, convert_element_type_default_51, convert_element_type_default_50, None, None, convert_element_type_197, None, convert_element_type_default_53, convert_element_type_default_52, None, None, convert_element_type_192, None, convert_element_type_default_55, convert_element_type_default_54, None, None, convert_element_type_187, None, convert_element_type_default_57, convert_element_type_default_56, None, None, convert_element_type_182, None, convert_element_type_default_59, convert_element_type_default_58, None, None, convert_element_type_177, None, convert_element_type_default_61, convert_element_type_default_60, None, None, convert_element_type_172, None, convert_element_type_default_63, convert_element_type_default_62, None, None, convert_element_type_167, None, convert_element_type_default_65, convert_element_type_default_64, None, None, convert_element_type_162, None, convert_element_type_default_67, convert_element_type_default_66, None, None, convert_element_type_157, None, convert_element_type_default_69, convert_element_type_default_68, None, None, convert_element_type_152, None, convert_element_type_default_71, convert_element_type_default_70, None, None, convert_element_type_147, None, convert_element_type_default_73, convert_element_type_default_72, None, None, convert_element_type_142, None, convert_element_type_default_75, convert_element_type_default_74, None, None, convert_element_type_137, None, convert_element_type_default_77, convert_element_type_default_76, None, None, convert_element_type_132, None, convert_element_type_default_79, convert_element_type_default_78, None, None, convert_element_type_127, None, convert_element_type_default_81, convert_element_type_default_80, None, None, convert_element_type_122, None, convert_element_type_default_83, convert_element_type_default_82, None, None, convert_element_type_117, None, convert_element_type_default_85, convert_element_type_default_84, None, None, convert_element_type_112, None, convert_element_type_default_87, convert_element_type_default_86, None, None, convert_element_type_107, None, convert_element_type_default_89, convert_element_type_default_88, None, None, convert_element_type_102, None, convert_element_type_default_91, convert_element_type_default_90, None, None, convert_element_type_97, None, convert_element_type_default_93, convert_element_type_default_92, None, None, convert_element_type_92, None, convert_element_type_default_95, convert_element_type_default_94, None, None, convert_element_type_87, None, convert_element_type_default_97, convert_element_type_default_96, None, None, convert_element_type_82, None, convert_element_type_default_99, convert_element_type_default_98, None, None, convert_element_type_77, None, convert_element_type_default_101, convert_element_type_default_100, None, None, convert_element_type_72, None, convert_element_type_default_103, convert_element_type_default_102, None, None, convert_element_type_67, None, convert_element_type_default_105, convert_element_type_default_104, None, None, convert_element_type_62)
        
def load_args(reader):
    buf0 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64,), is_leaf=True)  # primals_4
    buf1 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64,), is_leaf=True)  # primals_10
    buf2 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64,), is_leaf=True)  # primals_16
    buf3 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf3, (256,), is_leaf=True)  # primals_22
    buf4 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf4, (256,), is_leaf=True)  # primals_28
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64,), is_leaf=True)  # primals_34
    buf6 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64,), is_leaf=True)  # primals_40
    buf7 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf7, (256,), is_leaf=True)  # primals_46
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # primals_52
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # primals_58
    buf10 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf10, (256,), is_leaf=True)  # primals_64
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128,), is_leaf=True)  # primals_70
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # primals_76
    buf13 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf13, (512,), is_leaf=True)  # primals_82
    buf14 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf14, (512,), is_leaf=True)  # primals_88
    buf15 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf15, (128,), is_leaf=True)  # primals_94
    buf16 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128,), is_leaf=True)  # primals_100
    buf17 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf17, (512,), is_leaf=True)  # primals_106
    buf18 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf18, (128,), is_leaf=True)  # primals_112
    buf19 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128,), is_leaf=True)  # primals_118
    buf20 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf20, (512,), is_leaf=True)  # primals_124
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # primals_130
    buf22 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf22, (128,), is_leaf=True)  # primals_136
    buf23 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf23, (512,), is_leaf=True)  # primals_142
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256,), is_leaf=True)  # primals_148
    buf25 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256,), is_leaf=True)  # primals_154
    buf26 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1024,), is_leaf=True)  # primals_160
    buf27 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf27, (1024,), is_leaf=True)  # primals_166
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256,), is_leaf=True)  # primals_172
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256,), is_leaf=True)  # primals_178
    buf30 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1024,), is_leaf=True)  # primals_184
    buf31 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256,), is_leaf=True)  # primals_190
    buf32 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (256,), is_leaf=True)  # primals_196
    buf33 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf33, (1024,), is_leaf=True)  # primals_202
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # primals_208
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256,), is_leaf=True)  # primals_214
    buf36 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1024,), is_leaf=True)  # primals_220
    buf37 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256,), is_leaf=True)  # primals_226
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # primals_232
    buf39 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf39, (1024,), is_leaf=True)  # primals_238
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256,), is_leaf=True)  # primals_244
    buf41 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256,), is_leaf=True)  # primals_250
    buf42 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf42, (1024,), is_leaf=True)  # primals_256
    buf43 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512,), is_leaf=True)  # primals_262
    buf44 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf44, (512,), is_leaf=True)  # primals_268
    buf45 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf45, (2048,), is_leaf=True)  # primals_274
    buf46 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf46, (2048,), is_leaf=True)  # primals_280
    buf47 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512,), is_leaf=True)  # primals_286
    buf48 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf48, (512,), is_leaf=True)  # primals_292
    buf49 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf49, (2048,), is_leaf=True)  # primals_298
    buf50 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512,), is_leaf=True)  # primals_304
    buf51 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf51, (512,), is_leaf=True)  # primals_310
    buf52 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf52, (2048,), is_leaf=True)  # primals_316
    buf53 = reader.storage(None, 19267584, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf53, (150528, 32), dtype=torch.int32, is_leaf=True)  # getitem
    buf54 = reader.storage(None, 301056, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf54, (150528,), dtype=torch.bfloat16, is_leaf=True)  # getitem_1
    buf55 = reader.storage(None, 301056, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf55, (150528,), dtype=torch.bfloat16, is_leaf=True)  # getitem_2
    buf56 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf56, (64,), is_leaf=True)  # rsqrt
    buf57 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf57, (802816, 32), dtype=torch.int32, is_leaf=True)  # getitem_7
    buf58 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf58, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_8
    buf59 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf59, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_9
    buf60 = reader.storage(None, 822083584, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf60, (512, 64, 112, 112), dtype=torch.bfloat16, is_leaf=True)  # getitem_10
    buf61 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf61, (802816, 16), dtype=torch.int32, is_leaf=True)  # getitem_12
    buf62 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int8)
    reader.tensor(buf62, (512, 64, 56, 56), (200704, 1, 3584, 64), dtype=torch.int8, is_leaf=True)  # getitem_14
    buf63 = reader.storage(None, 8192, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf63, (64, 64, 1, 1), (64, 1, 64, 64), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_2
    buf64 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf64, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_15
    buf65 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf65, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_16
    buf66 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf66, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_17
    buf67 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf67, (64,), is_leaf=True)  # rsqrt_1
    buf68 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf68, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_22
    buf69 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf69, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_23
    buf70 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf70, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_24
    buf71 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf71, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_27
    buf72 = reader.storage(None, 73728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf72, (64, 64, 3, 3), (576, 1, 192, 64), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_3
    buf73 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf73, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_28
    buf74 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf74, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_29
    buf75 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf75, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_30
    buf76 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf76, (64,), is_leaf=True)  # rsqrt_2
    buf77 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf77, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_35
    buf78 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf78, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_36
    buf79 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf79, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_37
    buf80 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf80, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_40
    buf81 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf81, (256, 64, 1, 1), (64, 1, 64, 64), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_4
    buf82 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf82, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_41
    buf83 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf83, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_42
    buf84 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf84, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_43
    buf85 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf85, (256,), is_leaf=True)  # rsqrt_3
    buf86 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf86, (802816, 32), dtype=torch.int32, is_leaf=True)  # getitem_48
    buf87 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf87, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_49
    buf88 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf88, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_50
    buf89 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf89, (256, 64, 1, 1), (64, 1, 64, 64), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_5
    buf90 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf90, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_51
    buf91 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf91, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_52
    buf92 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf92, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_53
    buf93 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf93, (256,), is_leaf=True)  # rsqrt_4
    buf94 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf94, (802816, 32), dtype=torch.int32, is_leaf=True)  # getitem_58
    buf95 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf95, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_59
    buf96 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf96, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_60
    buf97 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf97, (802816, 16), dtype=torch.int32, is_leaf=True)  # getitem_63
    buf98 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf98, (64, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_6
    buf99 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf99, (802816, 32), dtype=torch.int32, is_leaf=True)  # getitem_64
    buf100 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf100, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_65
    buf101 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf101, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_66
    buf102 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf102, (64,), is_leaf=True)  # rsqrt_5
    buf103 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf103, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_71
    buf104 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf104, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_72
    buf105 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf105, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_73
    buf106 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf106, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_76
    buf107 = reader.storage(None, 73728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf107, (64, 64, 3, 3), (576, 1, 192, 64), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_7
    buf108 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf108, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_77
    buf109 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf109, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_78
    buf110 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf110, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_79
    buf111 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf111, (64,), is_leaf=True)  # rsqrt_6
    buf112 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf112, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_84
    buf113 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf113, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_85
    buf114 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf114, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_86
    buf115 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf115, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_89
    buf116 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf116, (256, 64, 1, 1), (64, 1, 64, 64), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_8
    buf117 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf117, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_90
    buf118 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf118, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_91
    buf119 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf119, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_92
    buf120 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf120, (256,), is_leaf=True)  # rsqrt_7
    buf121 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf121, (802816, 32), dtype=torch.int32, is_leaf=True)  # getitem_97
    buf122 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf122, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_98
    buf123 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf123, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_99
    buf124 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf124, (802816, 16), dtype=torch.int32, is_leaf=True)  # getitem_102
    buf125 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf125, (64, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_9
    buf126 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf126, (802816, 32), dtype=torch.int32, is_leaf=True)  # getitem_103
    buf127 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf127, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_104
    buf128 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf128, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_105
    buf129 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf129, (64,), is_leaf=True)  # rsqrt_8
    buf130 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf130, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_110
    buf131 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf131, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_111
    buf132 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf132, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_112
    buf133 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf133, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_115
    buf134 = reader.storage(None, 73728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf134, (64, 64, 3, 3), (576, 1, 192, 64), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_10
    buf135 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf135, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_116
    buf136 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf136, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_117
    buf137 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf137, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_118
    buf138 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf138, (64,), is_leaf=True)  # rsqrt_9
    buf139 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf139, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_123
    buf140 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf140, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_124
    buf141 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf141, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_125
    buf142 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf142, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_128
    buf143 = reader.storage(None, 32768, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf143, (256, 64, 1, 1), (64, 1, 64, 64), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_11
    buf144 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf144, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_129
    buf145 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf145, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_130
    buf146 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf146, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_131
    buf147 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf147, (256,), is_leaf=True)  # rsqrt_10
    buf148 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf148, (802816, 32), dtype=torch.int32, is_leaf=True)  # getitem_136
    buf149 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf149, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_137
    buf150 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf150, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_138
    buf151 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf151, (802816, 16), dtype=torch.int32, is_leaf=True)  # getitem_141
    buf152 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf152, (128, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_12
    buf153 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf153, (802816, 32), dtype=torch.int32, is_leaf=True)  # getitem_142
    buf154 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf154, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_143
    buf155 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf155, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_144
    buf156 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf156, (128,), is_leaf=True)  # rsqrt_11
    buf157 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf157, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_149
    buf158 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf158, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_150
    buf159 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf159, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_151
    buf160 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf160, (401408, 16), dtype=torch.int32, is_leaf=True)  # getitem_154
    buf161 = reader.storage(None, 294912, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf161, (128, 128, 3, 3), (1152, 1, 384, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_13
    buf162 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf162, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_155
    buf163 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf163, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_156
    buf164 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf164, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_157
    buf165 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf165, (128,), is_leaf=True)  # rsqrt_12
    buf166 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf166, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_162
    buf167 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf167, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_163
    buf168 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf168, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_164
    buf169 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf169, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_167
    buf170 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf170, (512, 128, 1, 1), (128, 1, 128, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_14
    buf171 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf171, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_168
    buf172 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf172, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_169
    buf173 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf173, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_170
    buf174 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf174, (512,), is_leaf=True)  # rsqrt_13
    buf175 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf175, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_175
    buf176 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf176, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_176
    buf177 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf177, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_177
    buf178 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf178, (512, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_15
    buf179 = reader.storage(None, 102760448, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf179, (802816, 32), dtype=torch.int32, is_leaf=True)  # getitem_178
    buf180 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf180, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_179
    buf181 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf181, (802816,), dtype=torch.bfloat16, is_leaf=True)  # getitem_180
    buf182 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf182, (512,), is_leaf=True)  # rsqrt_14
    buf183 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf183, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_185
    buf184 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf184, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_186
    buf185 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf185, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_187
    buf186 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf186, (401408, 16), dtype=torch.int32, is_leaf=True)  # getitem_190
    buf187 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf187, (128, 512, 1, 1), (512, 1, 512, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_16
    buf188 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf188, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_191
    buf189 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf189, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_192
    buf190 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf190, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_193
    buf191 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf191, (128,), is_leaf=True)  # rsqrt_15
    buf192 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf192, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_198
    buf193 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf193, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_199
    buf194 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf194, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_200
    buf195 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf195, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_203
    buf196 = reader.storage(None, 294912, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf196, (128, 128, 3, 3), (1152, 1, 384, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_17
    buf197 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf197, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_204
    buf198 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf198, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_205
    buf199 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf199, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_206
    buf200 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf200, (128,), is_leaf=True)  # rsqrt_16
    buf201 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf201, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_211
    buf202 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf202, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_212
    buf203 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf203, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_213
    buf204 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf204, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_216
    buf205 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf205, (512, 128, 1, 1), (128, 1, 128, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_18
    buf206 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf206, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_217
    buf207 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf207, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_218
    buf208 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf208, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_219
    buf209 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf209, (512,), is_leaf=True)  # rsqrt_17
    buf210 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf210, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_224
    buf211 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf211, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_225
    buf212 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf212, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_226
    buf213 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf213, (401408, 16), dtype=torch.int32, is_leaf=True)  # getitem_229
    buf214 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf214, (128, 512, 1, 1), (512, 1, 512, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_19
    buf215 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf215, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_230
    buf216 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf216, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_231
    buf217 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf217, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_232
    buf218 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf218, (128,), is_leaf=True)  # rsqrt_18
    buf219 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf219, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_237
    buf220 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf220, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_238
    buf221 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf221, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_239
    buf222 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf222, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_242
    buf223 = reader.storage(None, 294912, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf223, (128, 128, 3, 3), (1152, 1, 384, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_20
    buf224 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf224, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_243
    buf225 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf225, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_244
    buf226 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf226, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_245
    buf227 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf227, (128,), is_leaf=True)  # rsqrt_19
    buf228 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf228, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_250
    buf229 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf229, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_251
    buf230 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf230, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_252
    buf231 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf231, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_255
    buf232 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf232, (512, 128, 1, 1), (128, 1, 128, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_21
    buf233 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf233, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_256
    buf234 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf234, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_257
    buf235 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf235, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_258
    buf236 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf236, (512,), is_leaf=True)  # rsqrt_20
    buf237 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf237, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_263
    buf238 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf238, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_264
    buf239 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf239, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_265
    buf240 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf240, (401408, 16), dtype=torch.int32, is_leaf=True)  # getitem_268
    buf241 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf241, (128, 512, 1, 1), (512, 1, 512, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_22
    buf242 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf242, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_269
    buf243 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf243, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_270
    buf244 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf244, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_271
    buf245 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf245, (128,), is_leaf=True)  # rsqrt_21
    buf246 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf246, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_276
    buf247 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf247, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_277
    buf248 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf248, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_278
    buf249 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf249, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_281
    buf250 = reader.storage(None, 294912, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf250, (128, 128, 3, 3), (1152, 1, 384, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_23
    buf251 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf251, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_282
    buf252 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf252, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_283
    buf253 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf253, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_284
    buf254 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf254, (128,), is_leaf=True)  # rsqrt_22
    buf255 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf255, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_289
    buf256 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf256, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_290
    buf257 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf257, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_291
    buf258 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf258, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_294
    buf259 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf259, (512, 128, 1, 1), (128, 1, 128, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_24
    buf260 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf260, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_295
    buf261 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf261, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_296
    buf262 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf262, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_297
    buf263 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf263, (512,), is_leaf=True)  # rsqrt_23
    buf264 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf264, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_302
    buf265 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf265, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_303
    buf266 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf266, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_304
    buf267 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf267, (401408, 16), dtype=torch.int32, is_leaf=True)  # getitem_307
    buf268 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf268, (256, 512, 1, 1), (512, 1, 512, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_25
    buf269 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf269, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_308
    buf270 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf270, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_309
    buf271 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf271, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_310
    buf272 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf272, (256,), is_leaf=True)  # rsqrt_24
    buf273 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf273, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_315
    buf274 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf274, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_316
    buf275 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf275, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_317
    buf276 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf276, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_320
    buf277 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf277, (256, 256, 3, 3), (2304, 1, 768, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_26
    buf278 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf278, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_321
    buf279 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf279, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_322
    buf280 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf280, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_323
    buf281 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf281, (256,), is_leaf=True)  # rsqrt_25
    buf282 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf282, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_328
    buf283 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf283, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_329
    buf284 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf284, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_330
    buf285 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf285, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_333
    buf286 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf286, (1024, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_27
    buf287 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf287, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_334
    buf288 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf288, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_335
    buf289 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf289, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_336
    buf290 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf290, (1024,), is_leaf=True)  # rsqrt_26
    buf291 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf291, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_341
    buf292 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf292, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_342
    buf293 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf293, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_343
    buf294 = reader.storage(None, 1048576, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf294, (1024, 512, 1, 1), (512, 1, 512, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_28
    buf295 = reader.storage(None, 51380224, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf295, (401408, 32), dtype=torch.int32, is_leaf=True)  # getitem_344
    buf296 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf296, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_345
    buf297 = reader.storage(None, 802816, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf297, (401408,), dtype=torch.bfloat16, is_leaf=True)  # getitem_346
    buf298 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf298, (1024,), is_leaf=True)  # rsqrt_27
    buf299 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf299, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_351
    buf300 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf300, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_352
    buf301 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf301, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_353
    buf302 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf302, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_356
    buf303 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf303, (256, 1024, 1, 1), (1024, 1, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_29
    buf304 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf304, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_357
    buf305 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf305, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_358
    buf306 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf306, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_359
    buf307 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf307, (256,), is_leaf=True)  # rsqrt_28
    buf308 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf308, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_364
    buf309 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf309, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_365
    buf310 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf310, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_366
    buf311 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf311, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_369
    buf312 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf312, (256, 256, 3, 3), (2304, 1, 768, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_30
    buf313 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf313, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_370
    buf314 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf314, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_371
    buf315 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf315, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_372
    buf316 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf316, (256,), is_leaf=True)  # rsqrt_29
    buf317 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf317, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_377
    buf318 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf318, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_378
    buf319 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf319, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_379
    buf320 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf320, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_382
    buf321 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf321, (1024, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_31
    buf322 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf322, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_383
    buf323 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf323, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_384
    buf324 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf324, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_385
    buf325 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf325, (1024,), is_leaf=True)  # rsqrt_30
    buf326 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf326, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_390
    buf327 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf327, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_391
    buf328 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf328, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_392
    buf329 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf329, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_395
    buf330 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf330, (256, 1024, 1, 1), (1024, 1, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_32
    buf331 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf331, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_396
    buf332 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf332, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_397
    buf333 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf333, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_398
    buf334 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf334, (256,), is_leaf=True)  # rsqrt_31
    buf335 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf335, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_403
    buf336 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf336, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_404
    buf337 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf337, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_405
    buf338 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf338, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_408
    buf339 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf339, (256, 256, 3, 3), (2304, 1, 768, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_33
    buf340 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf340, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_409
    buf341 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf341, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_410
    buf342 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf342, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_411
    buf343 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf343, (256,), is_leaf=True)  # rsqrt_32
    buf344 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf344, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_416
    buf345 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf345, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_417
    buf346 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf346, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_418
    buf347 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf347, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_421
    buf348 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf348, (1024, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_34
    buf349 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf349, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_422
    buf350 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf350, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_423
    buf351 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf351, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_424
    buf352 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf352, (1024,), is_leaf=True)  # rsqrt_33
    buf353 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf353, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_429
    buf354 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf354, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_430
    buf355 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf355, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_431
    buf356 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf356, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_434
    buf357 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf357, (256, 1024, 1, 1), (1024, 1, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_35
    buf358 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf358, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_435
    buf359 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf359, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_436
    buf360 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf360, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_437
    buf361 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf361, (256,), is_leaf=True)  # rsqrt_34
    buf362 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf362, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_442
    buf363 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf363, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_443
    buf364 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf364, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_444
    buf365 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf365, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_447
    buf366 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf366, (256, 256, 3, 3), (2304, 1, 768, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_36
    buf367 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf367, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_448
    buf368 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf368, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_449
    buf369 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf369, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_450
    buf370 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf370, (256,), is_leaf=True)  # rsqrt_35
    buf371 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf371, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_455
    buf372 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf372, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_456
    buf373 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf373, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_457
    buf374 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf374, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_460
    buf375 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf375, (1024, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_37
    buf376 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf376, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_461
    buf377 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf377, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_462
    buf378 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf378, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_463
    buf379 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf379, (1024,), is_leaf=True)  # rsqrt_36
    buf380 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf380, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_468
    buf381 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf381, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_469
    buf382 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf382, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_470
    buf383 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf383, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_473
    buf384 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf384, (256, 1024, 1, 1), (1024, 1, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_38
    buf385 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf385, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_474
    buf386 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf386, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_475
    buf387 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf387, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_476
    buf388 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf388, (256,), is_leaf=True)  # rsqrt_37
    buf389 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf389, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_481
    buf390 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf390, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_482
    buf391 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf391, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_483
    buf392 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf392, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_486
    buf393 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf393, (256, 256, 3, 3), (2304, 1, 768, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_39
    buf394 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf394, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_487
    buf395 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf395, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_488
    buf396 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf396, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_489
    buf397 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf397, (256,), is_leaf=True)  # rsqrt_38
    buf398 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf398, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_494
    buf399 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf399, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_495
    buf400 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf400, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_496
    buf401 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf401, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_499
    buf402 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf402, (1024, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_40
    buf403 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf403, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_500
    buf404 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf404, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_501
    buf405 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf405, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_502
    buf406 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf406, (1024,), is_leaf=True)  # rsqrt_39
    buf407 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf407, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_507
    buf408 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf408, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_508
    buf409 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf409, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_509
    buf410 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf410, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_512
    buf411 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf411, (256, 1024, 1, 1), (1024, 1, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_41
    buf412 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf412, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_513
    buf413 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf413, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_514
    buf414 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf414, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_515
    buf415 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf415, (256,), is_leaf=True)  # rsqrt_40
    buf416 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf416, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_520
    buf417 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf417, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_521
    buf418 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf418, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_522
    buf419 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf419, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_525
    buf420 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf420, (256, 256, 3, 3), (2304, 1, 768, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_42
    buf421 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf421, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_526
    buf422 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf422, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_527
    buf423 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf423, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_528
    buf424 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf424, (256,), is_leaf=True)  # rsqrt_41
    buf425 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf425, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_533
    buf426 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf426, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_534
    buf427 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf427, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_535
    buf428 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf428, (50176, 16), dtype=torch.int32, is_leaf=True)  # getitem_538
    buf429 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf429, (1024, 256, 1, 1), (256, 1, 256, 256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_43
    buf430 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf430, (50176, 32), dtype=torch.int32, is_leaf=True)  # getitem_539
    buf431 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf431, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_540
    buf432 = reader.storage(None, 100352, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf432, (50176,), dtype=torch.bfloat16, is_leaf=True)  # getitem_541
    buf433 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf433, (1024,), is_leaf=True)  # rsqrt_42
    buf434 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf434, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_546
    buf435 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf435, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_547
    buf436 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf436, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_548
    buf437 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf437, (200704, 16), dtype=torch.int32, is_leaf=True)  # getitem_551
    buf438 = reader.storage(None, 1048576, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf438, (512, 1024, 1, 1), (1024, 1, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_44
    buf439 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf439, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_552
    buf440 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf440, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_553
    buf441 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf441, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_554
    buf442 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf442, (512,), is_leaf=True)  # rsqrt_43
    buf443 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf443, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_559
    buf444 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf444, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_560
    buf445 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf445, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_561
    buf446 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf446, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_564
    buf447 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf447, (512, 512, 3, 3), (4608, 1, 1536, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_45
    buf448 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf448, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_565
    buf449 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf449, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_566
    buf450 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf450, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_567
    buf451 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf451, (512,), is_leaf=True)  # rsqrt_44
    buf452 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf452, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_572
    buf453 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf453, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_573
    buf454 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf454, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_574
    buf455 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf455, (25088, 16), dtype=torch.int32, is_leaf=True)  # getitem_577
    buf456 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf456, (2048, 512, 1, 1), (512, 1, 512, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_46
    buf457 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf457, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_578
    buf458 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf458, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_579
    buf459 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf459, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_580
    buf460 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf460, (2048,), is_leaf=True)  # rsqrt_45
    buf461 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf461, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_585
    buf462 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf462, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_586
    buf463 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf463, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_587
    buf464 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf464, (2048, 1024, 1, 1), (1024, 1, 1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_47
    buf465 = reader.storage(None, 25690112, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf465, (200704, 32), dtype=torch.int32, is_leaf=True)  # getitem_588
    buf466 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf466, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_589
    buf467 = reader.storage(None, 401408, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf467, (200704,), dtype=torch.bfloat16, is_leaf=True)  # getitem_590
    buf468 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf468, (2048,), is_leaf=True)  # rsqrt_46
    buf469 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf469, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_595
    buf470 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf470, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_596
    buf471 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf471, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_597
    buf472 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf472, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_600
    buf473 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf473, (512, 2048, 1, 1), (2048, 1, 2048, 2048), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_48
    buf474 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf474, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_601
    buf475 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf475, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_602
    buf476 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf476, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_603
    buf477 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf477, (512,), is_leaf=True)  # rsqrt_47
    buf478 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf478, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_608
    buf479 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf479, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_609
    buf480 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf480, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_610
    buf481 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf481, (25088, 16), dtype=torch.int32, is_leaf=True)  # getitem_613
    buf482 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf482, (512, 512, 3, 3), (4608, 1, 1536, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_49
    buf483 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf483, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_614
    buf484 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf484, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_615
    buf485 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf485, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_616
    buf486 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf486, (512,), is_leaf=True)  # rsqrt_48
    buf487 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf487, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_621
    buf488 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf488, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_622
    buf489 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf489, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_623
    buf490 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf490, (25088, 16), dtype=torch.int32, is_leaf=True)  # getitem_626
    buf491 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf491, (2048, 512, 1, 1), (512, 1, 512, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_50
    buf492 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf492, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_627
    buf493 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf493, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_628
    buf494 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf494, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_629
    buf495 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf495, (2048,), is_leaf=True)  # rsqrt_49
    buf496 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf496, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_634
    buf497 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf497, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_635
    buf498 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf498, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_636
    buf499 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf499, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_639
    buf500 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf500, (512, 2048, 1, 1), (2048, 1, 2048, 2048), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_51
    buf501 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf501, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_640
    buf502 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf502, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_641
    buf503 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf503, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_642
    buf504 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf504, (512,), is_leaf=True)  # rsqrt_50
    buf505 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf505, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_647
    buf506 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf506, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_648
    buf507 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf507, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_649
    buf508 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf508, (25088, 16), dtype=torch.int32, is_leaf=True)  # getitem_652
    buf509 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf509, (512, 512, 3, 3), (4608, 1, 1536, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_52
    buf510 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf510, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_653
    buf511 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf511, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_654
    buf512 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf512, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_655
    buf513 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf513, (512,), is_leaf=True)  # rsqrt_51
    buf514 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf514, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_660
    buf515 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf515, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_661
    buf516 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf516, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_662
    buf517 = reader.storage(None, 1605632, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf517, (25088, 16), dtype=torch.int32, is_leaf=True)  # getitem_665
    buf518 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf518, (2048, 512, 1, 1), (512, 1, 512, 512), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_53
    buf519 = reader.storage(None, 3211264, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf519, (25088, 32), dtype=torch.int32, is_leaf=True)  # getitem_666
    buf520 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf520, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_667
    buf521 = reader.storage(None, 50176, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf521, (25088,), dtype=torch.bfloat16, is_leaf=True)  # getitem_668
    buf522 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf522, (2048,), is_leaf=True)  # rsqrt_52
    buf523 = reader.storage(None, 12845056, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf523, (100352, 32), dtype=torch.int32, is_leaf=True)  # getitem_673
    buf524 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf524, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_674
    buf525 = reader.storage(None, 200704, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf525, (100352,), dtype=torch.bfloat16, is_leaf=True)  # getitem_675
    buf526 = reader.storage(None, 6422528, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf526, (100352, 16), dtype=torch.int32, is_leaf=True)  # getitem_678
    buf527 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf527, (2048, 32), dtype=torch.int32, is_leaf=True)  # getitem_679
    buf528 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf528, (2048,), dtype=torch.bfloat16, is_leaf=True)  # getitem_680
    buf529 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf529, (2048,), dtype=torch.bfloat16, is_leaf=True)  # getitem_681
    buf530 = reader.storage(None, 409600, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf530, (100, 2048), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_54
    buf531 = reader.storage(None, 102400, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf531, (512, 100), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)