
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
torch._dynamo.config.verbose = True
torch._inductor.config.unroll_reductions_threshold = 8
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        convolution = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  convert_element_type_1 = convert_element_type = None
        add_10 = torch.ops.aten.add.Tensor(arg4_1, 1e-05);  arg4_1 = None
        sqrt = torch.ops.aten.sqrt.default(add_10);  add_10 = None
        reciprocal = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul_7 = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(mul_7, -1);  mul_7 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        sub_2 = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_3);  sub_2 = unsqueeze_3 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_11 = torch.ops.aten.add.Tensor(mul_9, unsqueeze_7);  mul_9 = unsqueeze_7 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        relu = torch.ops.aten.relu.default(convert_element_type_4);  convert_element_type_4 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convolution_1 = torch.ops.aten.convolution.default(getitem, convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_5 = None
        add_37 = torch.ops.aten.add.Tensor(arg9_1, 1e-05);  arg9_1 = None
        sqrt_1 = torch.ops.aten.sqrt.default(add_37);  add_37 = None
        reciprocal_1 = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_23 = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        sub_8 = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_11);  sub_8 = unsqueeze_11 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_38 = torch.ops.aten.add.Tensor(mul_25, unsqueeze_15);  mul_25 = unsqueeze_15 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None
        relu_1 = torch.ops.aten.relu.default(convert_element_type_8);  convert_element_type_8 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, convert_element_type_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = convert_element_type_9 = None
        add_54 = torch.ops.aten.add.Tensor(arg14_1, 1e-05);  arg14_1 = None
        sqrt_2 = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_2 = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_35 = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(mul_35, -1);  mul_35 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        sub_12 = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_19);  sub_12 = unsqueeze_19 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_21);  mul_36 = unsqueeze_21 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_55 = torch.ops.aten.add.Tensor(mul_37, unsqueeze_23);  mul_37 = unsqueeze_23 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        relu_2 = torch.ops.aten.relu.default(convert_element_type_12);  convert_element_type_12 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, convert_element_type_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = convert_element_type_13 = None
        add_71 = torch.ops.aten.add.Tensor(arg19_1, 1e-05);  arg19_1 = None
        sqrt_3 = torch.ops.aten.sqrt.default(add_71);  add_71 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_47 = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(mul_47, -1);  mul_47 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        sub_16 = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_27);  sub_16 = unsqueeze_27 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_29);  mul_48 = unsqueeze_29 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_72 = torch.ops.aten.add.Tensor(mul_49, unsqueeze_31);  mul_49 = unsqueeze_31 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        convolution_4 = torch.ops.aten.convolution.default(getitem, convert_element_type_17, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = convert_element_type_17 = None
        add_83 = torch.ops.aten.add.Tensor(arg24_1, 1e-05);  arg24_1 = None
        sqrt_4 = torch.ops.aten.sqrt.default(add_83);  add_83 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_57 = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        sub_19 = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_35);  sub_19 = unsqueeze_35 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_37);  mul_58 = unsqueeze_37 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_84 = torch.ops.aten.add.Tensor(mul_59, unsqueeze_39);  mul_59 = unsqueeze_39 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None
        add_100 = torch.ops.aten.add.Tensor(convert_element_type_16, convert_element_type_20);  convert_element_type_16 = convert_element_type_20 = None
        relu_3 = torch.ops.aten.relu.default(add_100);  add_100 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_3, convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_21 = None
        add_116 = torch.ops.aten.add.Tensor(arg29_1, 1e-05);  arg29_1 = None
        sqrt_5 = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_75 = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        sub_26 = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        mul_76 = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_43);  sub_26 = unsqueeze_43 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_45);  mul_76 = unsqueeze_45 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_117 = torch.ops.aten.add.Tensor(mul_77, unsqueeze_47);  mul_77 = unsqueeze_47 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None
        relu_4 = torch.ops.aten.relu.default(convert_element_type_24);  convert_element_type_24 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_4, convert_element_type_25, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = convert_element_type_25 = None
        add_133 = torch.ops.aten.add.Tensor(arg34_1, 1e-05);  arg34_1 = None
        sqrt_6 = torch.ops.aten.sqrt.default(add_133);  add_133 = None
        reciprocal_6 = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_87 = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        sub_30 = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_51);  sub_30 = unsqueeze_51 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_53);  mul_88 = unsqueeze_53 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_134 = torch.ops.aten.add.Tensor(mul_89, unsqueeze_55);  mul_89 = unsqueeze_55 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None
        relu_5 = torch.ops.aten.relu.default(convert_element_type_28);  convert_element_type_28 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_5, convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = convert_element_type_29 = None
        add_150 = torch.ops.aten.add.Tensor(arg39_1, 1e-05);  arg39_1 = None
        sqrt_7 = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_7 = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_99 = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        sub_34 = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_59);  sub_34 = unsqueeze_59 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_61);  mul_100 = unsqueeze_61 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_151 = torch.ops.aten.add.Tensor(mul_101, unsqueeze_63);  mul_101 = unsqueeze_63 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(add_151, torch.bfloat16);  add_151 = None
        add_167 = torch.ops.aten.add.Tensor(convert_element_type_32, relu_3);  convert_element_type_32 = relu_3 = None
        relu_6 = torch.ops.aten.relu.default(add_167);  add_167 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_6, convert_element_type_33, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_33 = None
        add_183 = torch.ops.aten.add.Tensor(arg44_1, 1e-05);  arg44_1 = None
        sqrt_8 = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_8 = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_117 = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        sub_41 = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        mul_118 = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_67);  sub_41 = unsqueeze_67 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_69);  mul_118 = unsqueeze_69 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_184 = torch.ops.aten.add.Tensor(mul_119, unsqueeze_71);  mul_119 = unsqueeze_71 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None
        relu_7 = torch.ops.aten.relu.default(convert_element_type_36);  convert_element_type_36 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_7, convert_element_type_37, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = convert_element_type_37 = None
        add_200 = torch.ops.aten.add.Tensor(arg49_1, 1e-05);  arg49_1 = None
        sqrt_9 = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_9 = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_129 = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        mul_130 = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_75);  sub_45 = unsqueeze_75 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_131 = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_77);  mul_130 = unsqueeze_77 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_201 = torch.ops.aten.add.Tensor(mul_131, unsqueeze_79);  mul_131 = unsqueeze_79 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(add_201, torch.bfloat16);  add_201 = None
        relu_8 = torch.ops.aten.relu.default(convert_element_type_40);  convert_element_type_40 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        convolution_10 = torch.ops.aten.convolution.default(relu_8, convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = convert_element_type_41 = None
        add_217 = torch.ops.aten.add.Tensor(arg54_1, 1e-05);  arg54_1 = None
        sqrt_10 = torch.ops.aten.sqrt.default(add_217);  add_217 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_141 = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        sub_49 = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        mul_142 = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_83);  sub_49 = unsqueeze_83 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_85);  mul_142 = unsqueeze_85 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_218 = torch.ops.aten.add.Tensor(mul_143, unsqueeze_87);  mul_143 = unsqueeze_87 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(add_218, torch.bfloat16);  add_218 = None
        add_234 = torch.ops.aten.add.Tensor(convert_element_type_44, relu_6);  convert_element_type_44 = relu_6 = None
        relu_9 = torch.ops.aten.relu.default(add_234);  add_234 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_9, convert_element_type_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_45 = None
        add_250 = torch.ops.aten.add.Tensor(arg59_1, 1e-05);  arg59_1 = None
        sqrt_11 = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_159 = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        sub_56 = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
        mul_160 = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_91);  sub_56 = unsqueeze_91 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_93);  mul_160 = unsqueeze_93 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_251 = torch.ops.aten.add.Tensor(mul_161, unsqueeze_95);  mul_161 = unsqueeze_95 = None
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(add_251, torch.bfloat16);  add_251 = None
        relu_10 = torch.ops.aten.relu.default(convert_element_type_48);  convert_element_type_48 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_10, convert_element_type_49, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = convert_element_type_49 = None
        add_267 = torch.ops.aten.add.Tensor(arg64_1, 1e-05);  arg64_1 = None
        sqrt_12 = torch.ops.aten.sqrt.default(add_267);  add_267 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_171 = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        sub_60 = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
        mul_172 = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_99);  sub_60 = unsqueeze_99 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_101);  mul_172 = unsqueeze_101 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_268 = torch.ops.aten.add.Tensor(mul_173, unsqueeze_103);  mul_173 = unsqueeze_103 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(add_268, torch.bfloat16);  add_268 = None
        relu_11 = torch.ops.aten.relu.default(convert_element_type_52);  convert_element_type_52 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_11, convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = convert_element_type_53 = None
        add_284 = torch.ops.aten.add.Tensor(arg69_1, 1e-05);  arg69_1 = None
        sqrt_13 = torch.ops.aten.sqrt.default(add_284);  add_284 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_183 = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        sub_64 = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
        mul_184 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_107);  sub_64 = unsqueeze_107 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_109);  mul_184 = unsqueeze_109 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_285 = torch.ops.aten.add.Tensor(mul_185, unsqueeze_111);  mul_185 = unsqueeze_111 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(add_285, torch.bfloat16);  add_285 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_9, convert_element_type_57, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = convert_element_type_57 = None
        add_296 = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_14 = torch.ops.aten.sqrt.default(add_296);  add_296 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_193 = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(mul_193, -1);  mul_193 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
        mul_194 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_115);  sub_67 = unsqueeze_115 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, unsqueeze_117);  mul_194 = unsqueeze_117 = None
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_297 = torch.ops.aten.add.Tensor(mul_195, unsqueeze_119);  mul_195 = unsqueeze_119 = None
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(add_297, torch.bfloat16);  add_297 = None
        add_313 = torch.ops.aten.add.Tensor(convert_element_type_56, convert_element_type_60);  convert_element_type_56 = convert_element_type_60 = None
        relu_12 = torch.ops.aten.relu.default(add_313);  add_313 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_12, convert_element_type_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_61 = None
        add_329 = torch.ops.aten.add.Tensor(arg79_1, 1e-05);  arg79_1 = None
        sqrt_15 = torch.ops.aten.sqrt.default(add_329);  add_329 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_211 = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(mul_211, -1);  mul_211 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        sub_74 = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
        mul_212 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_123);  sub_74 = unsqueeze_123 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, unsqueeze_125);  mul_212 = unsqueeze_125 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_330 = torch.ops.aten.add.Tensor(mul_213, unsqueeze_127);  mul_213 = unsqueeze_127 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(add_330, torch.bfloat16);  add_330 = None
        relu_13 = torch.ops.aten.relu.default(convert_element_type_64);  convert_element_type_64 = None
        convert_element_type_65 = torch.ops.prims.convert_element_type.default(arg82_1, torch.bfloat16);  arg82_1 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_13, convert_element_type_65, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = convert_element_type_65 = None
        add_346 = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
        sqrt_16 = torch.ops.aten.sqrt.default(add_346);  add_346 = None
        reciprocal_16 = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_223 = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(mul_223, -1);  mul_223 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        sub_78 = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
        mul_224 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_131);  sub_78 = unsqueeze_131 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_133);  mul_224 = unsqueeze_133 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_347 = torch.ops.aten.add.Tensor(mul_225, unsqueeze_135);  mul_225 = unsqueeze_135 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(add_347, torch.bfloat16);  add_347 = None
        relu_14 = torch.ops.aten.relu.default(convert_element_type_68);  convert_element_type_68 = None
        convert_element_type_69 = torch.ops.prims.convert_element_type.default(arg87_1, torch.bfloat16);  arg87_1 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_14, convert_element_type_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = convert_element_type_69 = None
        add_363 = torch.ops.aten.add.Tensor(arg89_1, 1e-05);  arg89_1 = None
        sqrt_17 = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_17 = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_235 = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(mul_235, -1);  mul_235 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        sub_82 = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
        mul_236 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_139);  sub_82 = unsqueeze_139 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_237 = torch.ops.aten.mul.Tensor(mul_236, unsqueeze_141);  mul_236 = unsqueeze_141 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_364 = torch.ops.aten.add.Tensor(mul_237, unsqueeze_143);  mul_237 = unsqueeze_143 = None
        convert_element_type_72 = torch.ops.prims.convert_element_type.default(add_364, torch.bfloat16);  add_364 = None
        add_380 = torch.ops.aten.add.Tensor(convert_element_type_72, relu_12);  convert_element_type_72 = relu_12 = None
        relu_15 = torch.ops.aten.relu.default(add_380);  add_380 = None
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(arg92_1, torch.bfloat16);  arg92_1 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_15, convert_element_type_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_73 = None
        add_396 = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
        sqrt_18 = torch.ops.aten.sqrt.default(add_396);  add_396 = None
        reciprocal_18 = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_253 = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        sub_89 = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_147);  sub_89 = unsqueeze_147 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_149);  mul_254 = unsqueeze_149 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_397 = torch.ops.aten.add.Tensor(mul_255, unsqueeze_151);  mul_255 = unsqueeze_151 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(add_397, torch.bfloat16);  add_397 = None
        relu_16 = torch.ops.aten.relu.default(convert_element_type_76);  convert_element_type_76 = None
        convert_element_type_77 = torch.ops.prims.convert_element_type.default(arg97_1, torch.bfloat16);  arg97_1 = None
        convolution_19 = torch.ops.aten.convolution.default(relu_16, convert_element_type_77, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = convert_element_type_77 = None
        add_413 = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
        sqrt_19 = torch.ops.aten.sqrt.default(add_413);  add_413 = None
        reciprocal_19 = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_265 = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(mul_265, -1);  mul_265 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
        mul_266 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_155);  sub_93 = unsqueeze_155 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_267 = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_157);  mul_266 = unsqueeze_157 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_414 = torch.ops.aten.add.Tensor(mul_267, unsqueeze_159);  mul_267 = unsqueeze_159 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(add_414, torch.bfloat16);  add_414 = None
        relu_17 = torch.ops.aten.relu.default(convert_element_type_80);  convert_element_type_80 = None
        convert_element_type_81 = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        convolution_20 = torch.ops.aten.convolution.default(relu_17, convert_element_type_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = convert_element_type_81 = None
        add_430 = torch.ops.aten.add.Tensor(arg104_1, 1e-05);  arg104_1 = None
        sqrt_20 = torch.ops.aten.sqrt.default(add_430);  add_430 = None
        reciprocal_20 = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_277 = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        sub_97 = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  convolution_20 = unsqueeze_161 = None
        mul_278 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_163);  sub_97 = unsqueeze_163 = None
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_165);  mul_278 = unsqueeze_165 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_431 = torch.ops.aten.add.Tensor(mul_279, unsqueeze_167);  mul_279 = unsqueeze_167 = None
        convert_element_type_84 = torch.ops.prims.convert_element_type.default(add_431, torch.bfloat16);  add_431 = None
        add_447 = torch.ops.aten.add.Tensor(convert_element_type_84, relu_15);  convert_element_type_84 = relu_15 = None
        relu_18 = torch.ops.aten.relu.default(add_447);  add_447 = None
        convert_element_type_85 = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        convolution_21 = torch.ops.aten.convolution.default(relu_18, convert_element_type_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_85 = None
        add_463 = torch.ops.aten.add.Tensor(arg109_1, 1e-05);  arg109_1 = None
        sqrt_21 = torch.ops.aten.sqrt.default(add_463);  add_463 = None
        reciprocal_21 = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_295 = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        sub_104 = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  convolution_21 = unsqueeze_169 = None
        mul_296 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_171);  sub_104 = unsqueeze_171 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_173);  mul_296 = unsqueeze_173 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_464 = torch.ops.aten.add.Tensor(mul_297, unsqueeze_175);  mul_297 = unsqueeze_175 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(add_464, torch.bfloat16);  add_464 = None
        relu_19 = torch.ops.aten.relu.default(convert_element_type_88);  convert_element_type_88 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        convolution_22 = torch.ops.aten.convolution.default(relu_19, convert_element_type_89, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_19 = convert_element_type_89 = None
        add_480 = torch.ops.aten.add.Tensor(arg114_1, 1e-05);  arg114_1 = None
        sqrt_22 = torch.ops.aten.sqrt.default(add_480);  add_480 = None
        reciprocal_22 = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_307 = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(mul_307, -1);  mul_307 = None
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        sub_108 = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  convolution_22 = unsqueeze_177 = None
        mul_308 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_179);  sub_108 = unsqueeze_179 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_309 = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_181);  mul_308 = unsqueeze_181 = None
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_481 = torch.ops.aten.add.Tensor(mul_309, unsqueeze_183);  mul_309 = unsqueeze_183 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(add_481, torch.bfloat16);  add_481 = None
        relu_20 = torch.ops.aten.relu.default(convert_element_type_92);  convert_element_type_92 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        convolution_23 = torch.ops.aten.convolution.default(relu_20, convert_element_type_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = convert_element_type_93 = None
        add_497 = torch.ops.aten.add.Tensor(arg119_1, 1e-05);  arg119_1 = None
        sqrt_23 = torch.ops.aten.sqrt.default(add_497);  add_497 = None
        reciprocal_23 = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_319 = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(mul_319, -1);  mul_319 = None
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        sub_112 = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  convolution_23 = unsqueeze_185 = None
        mul_320 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_187);  sub_112 = unsqueeze_187 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_320, unsqueeze_189);  mul_320 = unsqueeze_189 = None
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_498 = torch.ops.aten.add.Tensor(mul_321, unsqueeze_191);  mul_321 = unsqueeze_191 = None
        convert_element_type_96 = torch.ops.prims.convert_element_type.default(add_498, torch.bfloat16);  add_498 = None
        add_514 = torch.ops.aten.add.Tensor(convert_element_type_96, relu_18);  convert_element_type_96 = relu_18 = None
        relu_21 = torch.ops.aten.relu.default(add_514);  add_514 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        convolution_24 = torch.ops.aten.convolution.default(relu_21, convert_element_type_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_97 = None
        add_530 = torch.ops.aten.add.Tensor(arg124_1, 1e-05);  arg124_1 = None
        sqrt_24 = torch.ops.aten.sqrt.default(add_530);  add_530 = None
        reciprocal_24 = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_337 = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(mul_337, -1);  mul_337 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        sub_119 = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  convolution_24 = unsqueeze_193 = None
        mul_338 = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_195);  sub_119 = unsqueeze_195 = None
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_338, unsqueeze_197);  mul_338 = unsqueeze_197 = None
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_531 = torch.ops.aten.add.Tensor(mul_339, unsqueeze_199);  mul_339 = unsqueeze_199 = None
        convert_element_type_100 = torch.ops.prims.convert_element_type.default(add_531, torch.bfloat16);  add_531 = None
        relu_22 = torch.ops.aten.relu.default(convert_element_type_100);  convert_element_type_100 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        convolution_25 = torch.ops.aten.convolution.default(relu_22, convert_element_type_101, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_22 = convert_element_type_101 = None
        add_547 = torch.ops.aten.add.Tensor(arg129_1, 1e-05);  arg129_1 = None
        sqrt_25 = torch.ops.aten.sqrt.default(add_547);  add_547 = None
        reciprocal_25 = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_349 = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(mul_349, -1);  mul_349 = None
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        sub_123 = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  convolution_25 = unsqueeze_201 = None
        mul_350 = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_203);  sub_123 = unsqueeze_203 = None
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_351 = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_205);  mul_350 = unsqueeze_205 = None
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_548 = torch.ops.aten.add.Tensor(mul_351, unsqueeze_207);  mul_351 = unsqueeze_207 = None
        convert_element_type_104 = torch.ops.prims.convert_element_type.default(add_548, torch.bfloat16);  add_548 = None
        relu_23 = torch.ops.aten.relu.default(convert_element_type_104);  convert_element_type_104 = None
        convert_element_type_105 = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        convolution_26 = torch.ops.aten.convolution.default(relu_23, convert_element_type_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = convert_element_type_105 = None
        add_564 = torch.ops.aten.add.Tensor(arg134_1, 1e-05);  arg134_1 = None
        sqrt_26 = torch.ops.aten.sqrt.default(add_564);  add_564 = None
        reciprocal_26 = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_361 = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(mul_361, -1);  mul_361 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        sub_127 = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  convolution_26 = unsqueeze_209 = None
        mul_362 = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_211);  sub_127 = unsqueeze_211 = None
        unsqueeze_212 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_213 = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_363 = torch.ops.aten.mul.Tensor(mul_362, unsqueeze_213);  mul_362 = unsqueeze_213 = None
        unsqueeze_214 = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_565 = torch.ops.aten.add.Tensor(mul_363, unsqueeze_215);  mul_363 = unsqueeze_215 = None
        convert_element_type_108 = torch.ops.prims.convert_element_type.default(add_565, torch.bfloat16);  add_565 = None
        convert_element_type_109 = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        convolution_27 = torch.ops.aten.convolution.default(relu_21, convert_element_type_109, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = convert_element_type_109 = None
        add_576 = torch.ops.aten.add.Tensor(arg139_1, 1e-05);  arg139_1 = None
        sqrt_27 = torch.ops.aten.sqrt.default(add_576);  add_576 = None
        reciprocal_27 = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_371 = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(mul_371, -1);  mul_371 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        sub_130 = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  convolution_27 = unsqueeze_217 = None
        mul_372 = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_219);  sub_130 = unsqueeze_219 = None
        unsqueeze_220 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_221 = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_373 = torch.ops.aten.mul.Tensor(mul_372, unsqueeze_221);  mul_372 = unsqueeze_221 = None
        unsqueeze_222 = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_577 = torch.ops.aten.add.Tensor(mul_373, unsqueeze_223);  mul_373 = unsqueeze_223 = None
        convert_element_type_112 = torch.ops.prims.convert_element_type.default(add_577, torch.bfloat16);  add_577 = None
        add_593 = torch.ops.aten.add.Tensor(convert_element_type_108, convert_element_type_112);  convert_element_type_108 = convert_element_type_112 = None
        relu_24 = torch.ops.aten.relu.default(add_593);  add_593 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(arg142_1, torch.bfloat16);  arg142_1 = None
        convolution_28 = torch.ops.aten.convolution.default(relu_24, convert_element_type_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_113 = None
        add_609 = torch.ops.aten.add.Tensor(arg144_1, 1e-05);  arg144_1 = None
        sqrt_28 = torch.ops.aten.sqrt.default(add_609);  add_609 = None
        reciprocal_28 = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_389 = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(mul_389, -1);  mul_389 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        sub_137 = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  convolution_28 = unsqueeze_225 = None
        mul_390 = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_227);  sub_137 = unsqueeze_227 = None
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_391 = torch.ops.aten.mul.Tensor(mul_390, unsqueeze_229);  mul_390 = unsqueeze_229 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_610 = torch.ops.aten.add.Tensor(mul_391, unsqueeze_231);  mul_391 = unsqueeze_231 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(add_610, torch.bfloat16);  add_610 = None
        relu_25 = torch.ops.aten.relu.default(convert_element_type_116);  convert_element_type_116 = None
        convert_element_type_117 = torch.ops.prims.convert_element_type.default(arg147_1, torch.bfloat16);  arg147_1 = None
        convolution_29 = torch.ops.aten.convolution.default(relu_25, convert_element_type_117, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_25 = convert_element_type_117 = None
        add_626 = torch.ops.aten.add.Tensor(arg149_1, 1e-05);  arg149_1 = None
        sqrt_29 = torch.ops.aten.sqrt.default(add_626);  add_626 = None
        reciprocal_29 = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_401 = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(mul_401, -1);  mul_401 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        sub_141 = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  convolution_29 = unsqueeze_233 = None
        mul_402 = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_235);  sub_141 = unsqueeze_235 = None
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_403 = torch.ops.aten.mul.Tensor(mul_402, unsqueeze_237);  mul_402 = unsqueeze_237 = None
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_627 = torch.ops.aten.add.Tensor(mul_403, unsqueeze_239);  mul_403 = unsqueeze_239 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(add_627, torch.bfloat16);  add_627 = None
        relu_26 = torch.ops.aten.relu.default(convert_element_type_120);  convert_element_type_120 = None
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(arg152_1, torch.bfloat16);  arg152_1 = None
        convolution_30 = torch.ops.aten.convolution.default(relu_26, convert_element_type_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_26 = convert_element_type_121 = None
        add_643 = torch.ops.aten.add.Tensor(arg154_1, 1e-05);  arg154_1 = None
        sqrt_30 = torch.ops.aten.sqrt.default(add_643);  add_643 = None
        reciprocal_30 = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_413 = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(mul_413, -1);  mul_413 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        sub_145 = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  convolution_30 = unsqueeze_241 = None
        mul_414 = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_243);  sub_145 = unsqueeze_243 = None
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_415 = torch.ops.aten.mul.Tensor(mul_414, unsqueeze_245);  mul_414 = unsqueeze_245 = None
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_644 = torch.ops.aten.add.Tensor(mul_415, unsqueeze_247);  mul_415 = unsqueeze_247 = None
        convert_element_type_124 = torch.ops.prims.convert_element_type.default(add_644, torch.bfloat16);  add_644 = None
        add_660 = torch.ops.aten.add.Tensor(convert_element_type_124, relu_24);  convert_element_type_124 = relu_24 = None
        relu_27 = torch.ops.aten.relu.default(add_660);  add_660 = None
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        convolution_31 = torch.ops.aten.convolution.default(relu_27, convert_element_type_125, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_125 = None
        add_676 = torch.ops.aten.add.Tensor(arg159_1, 1e-05);  arg159_1 = None
        sqrt_31 = torch.ops.aten.sqrt.default(add_676);  add_676 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_431 = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(mul_431, -1);  mul_431 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        sub_152 = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  convolution_31 = unsqueeze_249 = None
        mul_432 = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_251);  sub_152 = unsqueeze_251 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_433 = torch.ops.aten.mul.Tensor(mul_432, unsqueeze_253);  mul_432 = unsqueeze_253 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_677 = torch.ops.aten.add.Tensor(mul_433, unsqueeze_255);  mul_433 = unsqueeze_255 = None
        convert_element_type_128 = torch.ops.prims.convert_element_type.default(add_677, torch.bfloat16);  add_677 = None
        relu_28 = torch.ops.aten.relu.default(convert_element_type_128);  convert_element_type_128 = None
        convert_element_type_129 = torch.ops.prims.convert_element_type.default(arg162_1, torch.bfloat16);  arg162_1 = None
        convolution_32 = torch.ops.aten.convolution.default(relu_28, convert_element_type_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_28 = convert_element_type_129 = None
        add_693 = torch.ops.aten.add.Tensor(arg164_1, 1e-05);  arg164_1 = None
        sqrt_32 = torch.ops.aten.sqrt.default(add_693);  add_693 = None
        reciprocal_32 = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_443 = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(mul_443, -1);  mul_443 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        sub_156 = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  convolution_32 = unsqueeze_257 = None
        mul_444 = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_259);  sub_156 = unsqueeze_259 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_445 = torch.ops.aten.mul.Tensor(mul_444, unsqueeze_261);  mul_444 = unsqueeze_261 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_694 = torch.ops.aten.add.Tensor(mul_445, unsqueeze_263);  mul_445 = unsqueeze_263 = None
        convert_element_type_132 = torch.ops.prims.convert_element_type.default(add_694, torch.bfloat16);  add_694 = None
        relu_29 = torch.ops.aten.relu.default(convert_element_type_132);  convert_element_type_132 = None
        convert_element_type_133 = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        convolution_33 = torch.ops.aten.convolution.default(relu_29, convert_element_type_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_29 = convert_element_type_133 = None
        add_710 = torch.ops.aten.add.Tensor(arg169_1, 1e-05);  arg169_1 = None
        sqrt_33 = torch.ops.aten.sqrt.default(add_710);  add_710 = None
        reciprocal_33 = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_455 = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(mul_455, -1);  mul_455 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        sub_160 = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  convolution_33 = unsqueeze_265 = None
        mul_456 = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_267);  sub_160 = unsqueeze_267 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_457 = torch.ops.aten.mul.Tensor(mul_456, unsqueeze_269);  mul_456 = unsqueeze_269 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_711 = torch.ops.aten.add.Tensor(mul_457, unsqueeze_271);  mul_457 = unsqueeze_271 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(add_711, torch.bfloat16);  add_711 = None
        add_727 = torch.ops.aten.add.Tensor(convert_element_type_136, relu_27);  convert_element_type_136 = relu_27 = None
        relu_30 = torch.ops.aten.relu.default(add_727);  add_727 = None
        convert_element_type_137 = torch.ops.prims.convert_element_type.default(arg172_1, torch.bfloat16);  arg172_1 = None
        convolution_34 = torch.ops.aten.convolution.default(relu_30, convert_element_type_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_137 = None
        add_743 = torch.ops.aten.add.Tensor(arg174_1, 1e-05);  arg174_1 = None
        sqrt_34 = torch.ops.aten.sqrt.default(add_743);  add_743 = None
        reciprocal_34 = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_473 = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_272 = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(mul_473, -1);  mul_473 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        sub_167 = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  convolution_34 = unsqueeze_273 = None
        mul_474 = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_275);  sub_167 = unsqueeze_275 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_475 = torch.ops.aten.mul.Tensor(mul_474, unsqueeze_277);  mul_474 = unsqueeze_277 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_744 = torch.ops.aten.add.Tensor(mul_475, unsqueeze_279);  mul_475 = unsqueeze_279 = None
        convert_element_type_140 = torch.ops.prims.convert_element_type.default(add_744, torch.bfloat16);  add_744 = None
        relu_31 = torch.ops.aten.relu.default(convert_element_type_140);  convert_element_type_140 = None
        convert_element_type_141 = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        convolution_35 = torch.ops.aten.convolution.default(relu_31, convert_element_type_141, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_31 = convert_element_type_141 = None
        add_760 = torch.ops.aten.add.Tensor(arg179_1, 1e-05);  arg179_1 = None
        sqrt_35 = torch.ops.aten.sqrt.default(add_760);  add_760 = None
        reciprocal_35 = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_485 = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(mul_485, -1);  mul_485 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        sub_171 = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  convolution_35 = unsqueeze_281 = None
        mul_486 = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_283);  sub_171 = unsqueeze_283 = None
        unsqueeze_284 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_487 = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_285);  mul_486 = unsqueeze_285 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_761 = torch.ops.aten.add.Tensor(mul_487, unsqueeze_287);  mul_487 = unsqueeze_287 = None
        convert_element_type_144 = torch.ops.prims.convert_element_type.default(add_761, torch.bfloat16);  add_761 = None
        relu_32 = torch.ops.aten.relu.default(convert_element_type_144);  convert_element_type_144 = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        convolution_36 = torch.ops.aten.convolution.default(relu_32, convert_element_type_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_32 = convert_element_type_145 = None
        add_777 = torch.ops.aten.add.Tensor(arg184_1, 1e-05);  arg184_1 = None
        sqrt_36 = torch.ops.aten.sqrt.default(add_777);  add_777 = None
        reciprocal_36 = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_497 = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(mul_497, -1);  mul_497 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        sub_175 = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  convolution_36 = unsqueeze_289 = None
        mul_498 = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_291);  sub_175 = unsqueeze_291 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_499 = torch.ops.aten.mul.Tensor(mul_498, unsqueeze_293);  mul_498 = unsqueeze_293 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_778 = torch.ops.aten.add.Tensor(mul_499, unsqueeze_295);  mul_499 = unsqueeze_295 = None
        convert_element_type_148 = torch.ops.prims.convert_element_type.default(add_778, torch.bfloat16);  add_778 = None
        add_794 = torch.ops.aten.add.Tensor(convert_element_type_148, relu_30);  convert_element_type_148 = relu_30 = None
        relu_33 = torch.ops.aten.relu.default(add_794);  add_794 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(arg187_1, torch.bfloat16);  arg187_1 = None
        convolution_37 = torch.ops.aten.convolution.default(relu_33, convert_element_type_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_149 = None
        add_810 = torch.ops.aten.add.Tensor(arg189_1, 1e-05);  arg189_1 = None
        sqrt_37 = torch.ops.aten.sqrt.default(add_810);  add_810 = None
        reciprocal_37 = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_515 = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_296 = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(mul_515, -1);  mul_515 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        sub_182 = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  convolution_37 = unsqueeze_297 = None
        mul_516 = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_299);  sub_182 = unsqueeze_299 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_517 = torch.ops.aten.mul.Tensor(mul_516, unsqueeze_301);  mul_516 = unsqueeze_301 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_811 = torch.ops.aten.add.Tensor(mul_517, unsqueeze_303);  mul_517 = unsqueeze_303 = None
        convert_element_type_152 = torch.ops.prims.convert_element_type.default(add_811, torch.bfloat16);  add_811 = None
        relu_34 = torch.ops.aten.relu.default(convert_element_type_152);  convert_element_type_152 = None
        convert_element_type_153 = torch.ops.prims.convert_element_type.default(arg192_1, torch.bfloat16);  arg192_1 = None
        convolution_38 = torch.ops.aten.convolution.default(relu_34, convert_element_type_153, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_34 = convert_element_type_153 = None
        add_827 = torch.ops.aten.add.Tensor(arg194_1, 1e-05);  arg194_1 = None
        sqrt_38 = torch.ops.aten.sqrt.default(add_827);  add_827 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_527 = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(mul_527, -1);  mul_527 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_186 = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  convolution_38 = unsqueeze_305 = None
        mul_528 = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_307);  sub_186 = unsqueeze_307 = None
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_529 = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_309);  mul_528 = unsqueeze_309 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_828 = torch.ops.aten.add.Tensor(mul_529, unsqueeze_311);  mul_529 = unsqueeze_311 = None
        convert_element_type_156 = torch.ops.prims.convert_element_type.default(add_828, torch.bfloat16);  add_828 = None
        relu_35 = torch.ops.aten.relu.default(convert_element_type_156);  convert_element_type_156 = None
        convert_element_type_157 = torch.ops.prims.convert_element_type.default(arg197_1, torch.bfloat16);  arg197_1 = None
        convolution_39 = torch.ops.aten.convolution.default(relu_35, convert_element_type_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_35 = convert_element_type_157 = None
        add_844 = torch.ops.aten.add.Tensor(arg199_1, 1e-05);  arg199_1 = None
        sqrt_39 = torch.ops.aten.sqrt.default(add_844);  add_844 = None
        reciprocal_39 = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_539 = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_539, -1);  mul_539 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_190 = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  convolution_39 = unsqueeze_313 = None
        mul_540 = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_315);  sub_190 = unsqueeze_315 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_541 = torch.ops.aten.mul.Tensor(mul_540, unsqueeze_317);  mul_540 = unsqueeze_317 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(arg201_1, -1);  arg201_1 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_845 = torch.ops.aten.add.Tensor(mul_541, unsqueeze_319);  mul_541 = unsqueeze_319 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(add_845, torch.bfloat16);  add_845 = None
        add_861 = torch.ops.aten.add.Tensor(convert_element_type_160, relu_33);  convert_element_type_160 = relu_33 = None
        relu_36 = torch.ops.aten.relu.default(add_861);  add_861 = None
        convert_element_type_161 = torch.ops.prims.convert_element_type.default(arg202_1, torch.bfloat16);  arg202_1 = None
        convolution_40 = torch.ops.aten.convolution.default(relu_36, convert_element_type_161, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_161 = None
        add_877 = torch.ops.aten.add.Tensor(arg204_1, 1e-05);  arg204_1 = None
        sqrt_40 = torch.ops.aten.sqrt.default(add_877);  add_877 = None
        reciprocal_40 = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_557 = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(mul_557, -1);  mul_557 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_197 = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  convolution_40 = unsqueeze_321 = None
        mul_558 = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_323);  sub_197 = unsqueeze_323 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_559 = torch.ops.aten.mul.Tensor(mul_558, unsqueeze_325);  mul_558 = unsqueeze_325 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_878 = torch.ops.aten.add.Tensor(mul_559, unsqueeze_327);  mul_559 = unsqueeze_327 = None
        convert_element_type_164 = torch.ops.prims.convert_element_type.default(add_878, torch.bfloat16);  add_878 = None
        relu_37 = torch.ops.aten.relu.default(convert_element_type_164);  convert_element_type_164 = None
        convert_element_type_165 = torch.ops.prims.convert_element_type.default(arg207_1, torch.bfloat16);  arg207_1 = None
        convolution_41 = torch.ops.aten.convolution.default(relu_37, convert_element_type_165, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_37 = convert_element_type_165 = None
        add_894 = torch.ops.aten.add.Tensor(arg209_1, 1e-05);  arg209_1 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_894);  add_894 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_569 = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_569, -1);  mul_569 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_201 = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
        mul_570 = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_331);  sub_201 = unsqueeze_331 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_571 = torch.ops.aten.mul.Tensor(mul_570, unsqueeze_333);  mul_570 = unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_895 = torch.ops.aten.add.Tensor(mul_571, unsqueeze_335);  mul_571 = unsqueeze_335 = None
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(add_895, torch.bfloat16);  add_895 = None
        relu_38 = torch.ops.aten.relu.default(convert_element_type_168);  convert_element_type_168 = None
        convert_element_type_169 = torch.ops.prims.convert_element_type.default(arg212_1, torch.bfloat16);  arg212_1 = None
        convolution_42 = torch.ops.aten.convolution.default(relu_38, convert_element_type_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_38 = convert_element_type_169 = None
        add_911 = torch.ops.aten.add.Tensor(arg214_1, 1e-05);  arg214_1 = None
        sqrt_42 = torch.ops.aten.sqrt.default(add_911);  add_911 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_581 = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_581, -1);  mul_581 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_205 = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
        mul_582 = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_339);  sub_205 = unsqueeze_339 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_583 = torch.ops.aten.mul.Tensor(mul_582, unsqueeze_341);  mul_582 = unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_912 = torch.ops.aten.add.Tensor(mul_583, unsqueeze_343);  mul_583 = unsqueeze_343 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(add_912, torch.bfloat16);  add_912 = None
        add_928 = torch.ops.aten.add.Tensor(convert_element_type_172, relu_36);  convert_element_type_172 = relu_36 = None
        relu_39 = torch.ops.aten.relu.default(add_928);  add_928 = None
        convert_element_type_173 = torch.ops.prims.convert_element_type.default(arg217_1, torch.bfloat16);  arg217_1 = None
        convolution_43 = torch.ops.aten.convolution.default(relu_39, convert_element_type_173, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_173 = None
        add_944 = torch.ops.aten.add.Tensor(arg219_1, 1e-05);  arg219_1 = None
        sqrt_43 = torch.ops.aten.sqrt.default(add_944);  add_944 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_599 = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(mul_599, -1);  mul_599 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_212 = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
        mul_600 = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_347);  sub_212 = unsqueeze_347 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_601 = torch.ops.aten.mul.Tensor(mul_600, unsqueeze_349);  mul_600 = unsqueeze_349 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_945 = torch.ops.aten.add.Tensor(mul_601, unsqueeze_351);  mul_601 = unsqueeze_351 = None
        convert_element_type_176 = torch.ops.prims.convert_element_type.default(add_945, torch.bfloat16);  add_945 = None
        relu_40 = torch.ops.aten.relu.default(convert_element_type_176);  convert_element_type_176 = None
        convert_element_type_177 = torch.ops.prims.convert_element_type.default(arg222_1, torch.bfloat16);  arg222_1 = None
        convolution_44 = torch.ops.aten.convolution.default(relu_40, convert_element_type_177, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_40 = convert_element_type_177 = None
        add_961 = torch.ops.aten.add.Tensor(arg224_1, 1e-05);  arg224_1 = None
        sqrt_44 = torch.ops.aten.sqrt.default(add_961);  add_961 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_611 = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_611, -1);  mul_611 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_216 = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
        mul_612 = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_355);  sub_216 = unsqueeze_355 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_613 = torch.ops.aten.mul.Tensor(mul_612, unsqueeze_357);  mul_612 = unsqueeze_357 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_962 = torch.ops.aten.add.Tensor(mul_613, unsqueeze_359);  mul_613 = unsqueeze_359 = None
        convert_element_type_180 = torch.ops.prims.convert_element_type.default(add_962, torch.bfloat16);  add_962 = None
        relu_41 = torch.ops.aten.relu.default(convert_element_type_180);  convert_element_type_180 = None
        convert_element_type_181 = torch.ops.prims.convert_element_type.default(arg227_1, torch.bfloat16);  arg227_1 = None
        convolution_45 = torch.ops.aten.convolution.default(relu_41, convert_element_type_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_41 = convert_element_type_181 = None
        add_978 = torch.ops.aten.add.Tensor(arg229_1, 1e-05);  arg229_1 = None
        sqrt_45 = torch.ops.aten.sqrt.default(add_978);  add_978 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_623 = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_623, -1);  mul_623 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_220 = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
        mul_624 = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_363);  sub_220 = unsqueeze_363 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_625 = torch.ops.aten.mul.Tensor(mul_624, unsqueeze_365);  mul_624 = unsqueeze_365 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_979 = torch.ops.aten.add.Tensor(mul_625, unsqueeze_367);  mul_625 = unsqueeze_367 = None
        convert_element_type_184 = torch.ops.prims.convert_element_type.default(add_979, torch.bfloat16);  add_979 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(arg232_1, torch.bfloat16);  arg232_1 = None
        convolution_46 = torch.ops.aten.convolution.default(relu_39, convert_element_type_185, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_39 = convert_element_type_185 = None
        add_990 = torch.ops.aten.add.Tensor(arg234_1, 1e-05);  arg234_1 = None
        sqrt_46 = torch.ops.aten.sqrt.default(add_990);  add_990 = None
        reciprocal_46 = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_633 = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(mul_633, -1);  mul_633 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_223 = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
        mul_634 = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_371);  sub_223 = unsqueeze_371 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_635 = torch.ops.aten.mul.Tensor(mul_634, unsqueeze_373);  mul_634 = unsqueeze_373 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_991 = torch.ops.aten.add.Tensor(mul_635, unsqueeze_375);  mul_635 = unsqueeze_375 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(add_991, torch.bfloat16);  add_991 = None
        add_1007 = torch.ops.aten.add.Tensor(convert_element_type_184, convert_element_type_188);  convert_element_type_184 = convert_element_type_188 = None
        relu_42 = torch.ops.aten.relu.default(add_1007);  add_1007 = None
        convert_element_type_189 = torch.ops.prims.convert_element_type.default(arg237_1, torch.bfloat16);  arg237_1 = None
        convolution_47 = torch.ops.aten.convolution.default(relu_42, convert_element_type_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_189 = None
        add_1023 = torch.ops.aten.add.Tensor(arg239_1, 1e-05);  arg239_1 = None
        sqrt_47 = torch.ops.aten.sqrt.default(add_1023);  add_1023 = None
        reciprocal_47 = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_651 = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(mul_651, -1);  mul_651 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_230 = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
        mul_652 = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_379);  sub_230 = unsqueeze_379 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_653 = torch.ops.aten.mul.Tensor(mul_652, unsqueeze_381);  mul_652 = unsqueeze_381 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_1024 = torch.ops.aten.add.Tensor(mul_653, unsqueeze_383);  mul_653 = unsqueeze_383 = None
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(add_1024, torch.bfloat16);  add_1024 = None
        relu_43 = torch.ops.aten.relu.default(convert_element_type_192);  convert_element_type_192 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(arg242_1, torch.bfloat16);  arg242_1 = None
        convolution_48 = torch.ops.aten.convolution.default(relu_43, convert_element_type_193, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_43 = convert_element_type_193 = None
        add_1040 = torch.ops.aten.add.Tensor(arg244_1, 1e-05);  arg244_1 = None
        sqrt_48 = torch.ops.aten.sqrt.default(add_1040);  add_1040 = None
        reciprocal_48 = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_663 = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(mul_663, -1);  mul_663 = None
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_234 = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
        mul_664 = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_387);  sub_234 = unsqueeze_387 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_665 = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_389);  mul_664 = unsqueeze_389 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_1041 = torch.ops.aten.add.Tensor(mul_665, unsqueeze_391);  mul_665 = unsqueeze_391 = None
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(add_1041, torch.bfloat16);  add_1041 = None
        relu_44 = torch.ops.aten.relu.default(convert_element_type_196);  convert_element_type_196 = None
        convert_element_type_197 = torch.ops.prims.convert_element_type.default(arg247_1, torch.bfloat16);  arg247_1 = None
        convolution_49 = torch.ops.aten.convolution.default(relu_44, convert_element_type_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_44 = convert_element_type_197 = None
        add_1057 = torch.ops.aten.add.Tensor(arg249_1, 1e-05);  arg249_1 = None
        sqrt_49 = torch.ops.aten.sqrt.default(add_1057);  add_1057 = None
        reciprocal_49 = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_675 = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(mul_675, -1);  mul_675 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_238 = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
        mul_676 = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_395);  sub_238 = unsqueeze_395 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_677 = torch.ops.aten.mul.Tensor(mul_676, unsqueeze_397);  mul_676 = unsqueeze_397 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_1058 = torch.ops.aten.add.Tensor(mul_677, unsqueeze_399);  mul_677 = unsqueeze_399 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(add_1058, torch.bfloat16);  add_1058 = None
        add_1074 = torch.ops.aten.add.Tensor(convert_element_type_200, relu_42);  convert_element_type_200 = relu_42 = None
        relu_45 = torch.ops.aten.relu.default(add_1074);  add_1074 = None
        convert_element_type_201 = torch.ops.prims.convert_element_type.default(arg252_1, torch.bfloat16);  arg252_1 = None
        convolution_50 = torch.ops.aten.convolution.default(relu_45, convert_element_type_201, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_201 = None
        add_1090 = torch.ops.aten.add.Tensor(arg254_1, 1e-05);  arg254_1 = None
        sqrt_50 = torch.ops.aten.sqrt.default(add_1090);  add_1090 = None
        reciprocal_50 = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_693 = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_693, -1);  mul_693 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_245 = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
        mul_694 = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_403);  sub_245 = unsqueeze_403 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_695 = torch.ops.aten.mul.Tensor(mul_694, unsqueeze_405);  mul_694 = unsqueeze_405 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_1091 = torch.ops.aten.add.Tensor(mul_695, unsqueeze_407);  mul_695 = unsqueeze_407 = None
        convert_element_type_204 = torch.ops.prims.convert_element_type.default(add_1091, torch.bfloat16);  add_1091 = None
        relu_46 = torch.ops.aten.relu.default(convert_element_type_204);  convert_element_type_204 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(arg257_1, torch.bfloat16);  arg257_1 = None
        convolution_51 = torch.ops.aten.convolution.default(relu_46, convert_element_type_205, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_46 = convert_element_type_205 = None
        add_1107 = torch.ops.aten.add.Tensor(arg259_1, 1e-05);  arg259_1 = None
        sqrt_51 = torch.ops.aten.sqrt.default(add_1107);  add_1107 = None
        reciprocal_51 = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_705 = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(mul_705, -1);  mul_705 = None
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_249 = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
        mul_706 = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_411);  sub_249 = unsqueeze_411 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_707 = torch.ops.aten.mul.Tensor(mul_706, unsqueeze_413);  mul_706 = unsqueeze_413 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_1108 = torch.ops.aten.add.Tensor(mul_707, unsqueeze_415);  mul_707 = unsqueeze_415 = None
        convert_element_type_208 = torch.ops.prims.convert_element_type.default(add_1108, torch.bfloat16);  add_1108 = None
        relu_47 = torch.ops.aten.relu.default(convert_element_type_208);  convert_element_type_208 = None
        convert_element_type_209 = torch.ops.prims.convert_element_type.default(arg262_1, torch.bfloat16);  arg262_1 = None
        convolution_52 = torch.ops.aten.convolution.default(relu_47, convert_element_type_209, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_47 = convert_element_type_209 = None
        add_1124 = torch.ops.aten.add.Tensor(arg264_1, 1e-05);  arg264_1 = None
        sqrt_52 = torch.ops.aten.sqrt.default(add_1124);  add_1124 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_717 = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(mul_717, -1);  mul_717 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_253 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
        mul_718 = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_419);  sub_253 = unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_719 = torch.ops.aten.mul.Tensor(mul_718, unsqueeze_421);  mul_718 = unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_1125 = torch.ops.aten.add.Tensor(mul_719, unsqueeze_423);  mul_719 = unsqueeze_423 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(add_1125, torch.bfloat16);  add_1125 = None
        add_1141 = torch.ops.aten.add.Tensor(convert_element_type_212, relu_45);  convert_element_type_212 = relu_45 = None
        relu_48 = torch.ops.aten.relu.default(add_1141);  add_1141 = None
        mean = torch.ops.aten.mean.dim(relu_48, [-1, -2], True);  relu_48 = None
        view = torch.ops.aten.view.default(mean, [arg1_1, 2048]);  mean = arg1_1 = None
        convert_element_type_213 = torch.ops.prims.convert_element_type.default(arg268_1, torch.bfloat16);  arg268_1 = None
        convert_element_type_214 = torch.ops.prims.convert_element_type.default(arg267_1, torch.bfloat16);  arg267_1 = None
        permute = torch.ops.aten.permute.default(convert_element_type_214, [1, 0]);  convert_element_type_214 = None
        addmm = torch.ops.aten.addmm.default(convert_element_type_213, view, permute);  convert_element_type_213 = view = permute = None
        return (addmm,)
        
def load_args(reader):
    buf0 = reader.storage(None, 37632, device=device(type='cuda', index=0))
    reader.tensor(buf0, (64, 3, 7, 7), is_leaf=True)  # arg0_1
    reader.symint(64)  # arg1_1
    buf1 = reader.storage(None, 602112*s77, device=device(type='cuda', index=0))
    reader.tensor(buf1, (s77, 3, 224, 224), is_leaf=True)  # arg2_1
    buf2 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf2, (64,), is_leaf=True)  # arg3_1
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (64,), is_leaf=True)  # arg4_1
    buf4 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf4, (64,), is_leaf=True)  # arg5_1
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf5, (64,), is_leaf=True)  # arg6_1
    buf6 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 64, 1, 1), is_leaf=True)  # arg7_1
    buf7 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64,), is_leaf=True)  # arg8_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # arg9_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # arg10_1
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64,), is_leaf=True)  # arg11_1
    buf11 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64, 64, 3, 3), is_leaf=True)  # arg12_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # arg13_1
    buf13 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64,), is_leaf=True)  # arg14_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg15_1
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # arg16_1
    buf16 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf16, (256, 64, 1, 1), is_leaf=True)  # arg17_1
    buf17 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf17, (256,), is_leaf=True)  # arg18_1
    buf18 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf18, (256,), is_leaf=True)  # arg19_1
    buf19 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf19, (256,), is_leaf=True)  # arg20_1
    buf20 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf20, (256,), is_leaf=True)  # arg21_1
    buf21 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256, 64, 1, 1), is_leaf=True)  # arg22_1
    buf22 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256,), is_leaf=True)  # arg23_1
    buf23 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256,), is_leaf=True)  # arg24_1
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256,), is_leaf=True)  # arg25_1
    buf25 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256,), is_leaf=True)  # arg26_1
    buf26 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf26, (64, 256, 1, 1), is_leaf=True)  # arg27_1
    buf27 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf27, (64,), is_leaf=True)  # arg28_1
    buf28 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf28, (64,), is_leaf=True)  # arg29_1
    buf29 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf29, (64,), is_leaf=True)  # arg30_1
    buf30 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf30, (64,), is_leaf=True)  # arg31_1
    buf31 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64, 64, 3, 3), is_leaf=True)  # arg32_1
    buf32 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf32, (64,), is_leaf=True)  # arg33_1
    buf33 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf33, (64,), is_leaf=True)  # arg34_1
    buf34 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf34, (64,), is_leaf=True)  # arg35_1
    buf35 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf35, (64,), is_leaf=True)  # arg36_1
    buf36 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256, 64, 1, 1), is_leaf=True)  # arg37_1
    buf37 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256,), is_leaf=True)  # arg38_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg39_1
    buf39 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256,), is_leaf=True)  # arg40_1
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256,), is_leaf=True)  # arg41_1
    buf41 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf41, (64, 256, 1, 1), is_leaf=True)  # arg42_1
    buf42 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf42, (64,), is_leaf=True)  # arg43_1
    buf43 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf43, (64,), is_leaf=True)  # arg44_1
    buf44 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64,), is_leaf=True)  # arg45_1
    buf45 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64,), is_leaf=True)  # arg46_1
    buf46 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64, 64, 3, 3), is_leaf=True)  # arg47_1
    buf47 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64,), is_leaf=True)  # arg48_1
    buf48 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # arg49_1
    buf49 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64,), is_leaf=True)  # arg50_1
    buf50 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf50, (64,), is_leaf=True)  # arg51_1
    buf51 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf51, (256, 64, 1, 1), is_leaf=True)  # arg52_1
    buf52 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf52, (256,), is_leaf=True)  # arg53_1
    buf53 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf53, (256,), is_leaf=True)  # arg54_1
    buf54 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf54, (256,), is_leaf=True)  # arg55_1
    buf55 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256,), is_leaf=True)  # arg56_1
    buf56 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (128, 256, 1, 1), is_leaf=True)  # arg57_1
    buf57 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf57, (128,), is_leaf=True)  # arg58_1
    buf58 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf58, (128,), is_leaf=True)  # arg59_1
    buf59 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf59, (128,), is_leaf=True)  # arg60_1
    buf60 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf60, (128,), is_leaf=True)  # arg61_1
    buf61 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128, 128, 3, 3), is_leaf=True)  # arg62_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg63_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg64_1
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # arg65_1
    buf65 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf65, (128,), is_leaf=True)  # arg66_1
    buf66 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512, 128, 1, 1), is_leaf=True)  # arg67_1
    buf67 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512,), is_leaf=True)  # arg68_1
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # arg69_1
    buf69 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512,), is_leaf=True)  # arg70_1
    buf70 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512,), is_leaf=True)  # arg71_1
    buf71 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512, 256, 1, 1), is_leaf=True)  # arg72_1
    buf72 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf72, (512,), is_leaf=True)  # arg73_1
    buf73 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512,), is_leaf=True)  # arg74_1
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # arg75_1
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # arg76_1
    buf76 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf76, (128, 512, 1, 1), is_leaf=True)  # arg77_1
    buf77 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf77, (128,), is_leaf=True)  # arg78_1
    buf78 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf78, (128,), is_leaf=True)  # arg79_1
    buf79 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf79, (128,), is_leaf=True)  # arg80_1
    buf80 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf80, (128,), is_leaf=True)  # arg81_1
    buf81 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf81, (128, 128, 3, 3), is_leaf=True)  # arg82_1
    buf82 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf82, (128,), is_leaf=True)  # arg83_1
    buf83 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128,), is_leaf=True)  # arg84_1
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # arg85_1
    buf85 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf85, (128,), is_leaf=True)  # arg86_1
    buf86 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512, 128, 1, 1), is_leaf=True)  # arg87_1
    buf87 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512,), is_leaf=True)  # arg88_1
    buf88 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf88, (512,), is_leaf=True)  # arg89_1
    buf89 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf89, (512,), is_leaf=True)  # arg90_1
    buf90 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf90, (512,), is_leaf=True)  # arg91_1
    buf91 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf91, (128, 512, 1, 1), is_leaf=True)  # arg92_1
    buf92 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf92, (128,), is_leaf=True)  # arg93_1
    buf93 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf93, (128,), is_leaf=True)  # arg94_1
    buf94 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf94, (128,), is_leaf=True)  # arg95_1
    buf95 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf95, (128,), is_leaf=True)  # arg96_1
    buf96 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf96, (128, 128, 3, 3), is_leaf=True)  # arg97_1
    buf97 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf97, (128,), is_leaf=True)  # arg98_1
    buf98 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf98, (128,), is_leaf=True)  # arg99_1
    buf99 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf99, (128,), is_leaf=True)  # arg100_1
    buf100 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf100, (128,), is_leaf=True)  # arg101_1
    buf101 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf101, (512, 128, 1, 1), is_leaf=True)  # arg102_1
    buf102 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf102, (512,), is_leaf=True)  # arg103_1
    buf103 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf103, (512,), is_leaf=True)  # arg104_1
    buf104 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf104, (512,), is_leaf=True)  # arg105_1
    buf105 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf105, (512,), is_leaf=True)  # arg106_1
    buf106 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf106, (128, 512, 1, 1), is_leaf=True)  # arg107_1
    buf107 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf107, (128,), is_leaf=True)  # arg108_1
    buf108 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf108, (128,), is_leaf=True)  # arg109_1
    buf109 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf109, (128,), is_leaf=True)  # arg110_1
    buf110 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf110, (128,), is_leaf=True)  # arg111_1
    buf111 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf111, (128, 128, 3, 3), is_leaf=True)  # arg112_1
    buf112 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf112, (128,), is_leaf=True)  # arg113_1
    buf113 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf113, (128,), is_leaf=True)  # arg114_1
    buf114 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf114, (128,), is_leaf=True)  # arg115_1
    buf115 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf115, (128,), is_leaf=True)  # arg116_1
    buf116 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf116, (512, 128, 1, 1), is_leaf=True)  # arg117_1
    buf117 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf117, (512,), is_leaf=True)  # arg118_1
    buf118 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf118, (512,), is_leaf=True)  # arg119_1
    buf119 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf119, (512,), is_leaf=True)  # arg120_1
    buf120 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf120, (512,), is_leaf=True)  # arg121_1
    buf121 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256, 512, 1, 1), is_leaf=True)  # arg122_1
    buf122 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf122, (256,), is_leaf=True)  # arg123_1
    buf123 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256,), is_leaf=True)  # arg124_1
    buf124 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf124, (256,), is_leaf=True)  # arg125_1
    buf125 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf125, (256,), is_leaf=True)  # arg126_1
    buf126 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf126, (256, 256, 3, 3), is_leaf=True)  # arg127_1
    buf127 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf127, (256,), is_leaf=True)  # arg128_1
    buf128 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf128, (256,), is_leaf=True)  # arg129_1
    buf129 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf129, (256,), is_leaf=True)  # arg130_1
    buf130 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf130, (256,), is_leaf=True)  # arg131_1
    buf131 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf131, (1024, 256, 1, 1), is_leaf=True)  # arg132_1
    buf132 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf132, (1024,), is_leaf=True)  # arg133_1
    buf133 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1024,), is_leaf=True)  # arg134_1
    buf134 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf134, (1024,), is_leaf=True)  # arg135_1
    buf135 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1024,), is_leaf=True)  # arg136_1
    buf136 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1024, 512, 1, 1), is_leaf=True)  # arg137_1
    buf137 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1024,), is_leaf=True)  # arg138_1
    buf138 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf138, (1024,), is_leaf=True)  # arg139_1
    buf139 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1024,), is_leaf=True)  # arg140_1
    buf140 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1024,), is_leaf=True)  # arg141_1
    buf141 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf141, (256, 1024, 1, 1), is_leaf=True)  # arg142_1
    buf142 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf142, (256,), is_leaf=True)  # arg143_1
    buf143 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf143, (256,), is_leaf=True)  # arg144_1
    buf144 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf144, (256,), is_leaf=True)  # arg145_1
    buf145 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf145, (256,), is_leaf=True)  # arg146_1
    buf146 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf146, (256, 256, 3, 3), is_leaf=True)  # arg147_1
    buf147 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf147, (256,), is_leaf=True)  # arg148_1
    buf148 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf148, (256,), is_leaf=True)  # arg149_1
    buf149 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf149, (256,), is_leaf=True)  # arg150_1
    buf150 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf150, (256,), is_leaf=True)  # arg151_1
    buf151 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf151, (1024, 256, 1, 1), is_leaf=True)  # arg152_1
    buf152 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1024,), is_leaf=True)  # arg153_1
    buf153 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf153, (1024,), is_leaf=True)  # arg154_1
    buf154 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf154, (1024,), is_leaf=True)  # arg155_1
    buf155 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1024,), is_leaf=True)  # arg156_1
    buf156 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf156, (256, 1024, 1, 1), is_leaf=True)  # arg157_1
    buf157 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf157, (256,), is_leaf=True)  # arg158_1
    buf158 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf158, (256,), is_leaf=True)  # arg159_1
    buf159 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf159, (256,), is_leaf=True)  # arg160_1
    buf160 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf160, (256,), is_leaf=True)  # arg161_1
    buf161 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf161, (256, 256, 3, 3), is_leaf=True)  # arg162_1
    buf162 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf162, (256,), is_leaf=True)  # arg163_1
    buf163 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf163, (256,), is_leaf=True)  # arg164_1
    buf164 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf164, (256,), is_leaf=True)  # arg165_1
    buf165 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf165, (256,), is_leaf=True)  # arg166_1
    buf166 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf166, (1024, 256, 1, 1), is_leaf=True)  # arg167_1
    buf167 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf167, (1024,), is_leaf=True)  # arg168_1
    buf168 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1024,), is_leaf=True)  # arg169_1
    buf169 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1024,), is_leaf=True)  # arg170_1
    buf170 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf170, (1024,), is_leaf=True)  # arg171_1
    buf171 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf171, (256, 1024, 1, 1), is_leaf=True)  # arg172_1
    buf172 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf172, (256,), is_leaf=True)  # arg173_1
    buf173 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf173, (256,), is_leaf=True)  # arg174_1
    buf174 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf174, (256,), is_leaf=True)  # arg175_1
    buf175 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf175, (256,), is_leaf=True)  # arg176_1
    buf176 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf176, (256, 256, 3, 3), is_leaf=True)  # arg177_1
    buf177 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf177, (256,), is_leaf=True)  # arg178_1
    buf178 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf178, (256,), is_leaf=True)  # arg179_1
    buf179 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf179, (256,), is_leaf=True)  # arg180_1
    buf180 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf180, (256,), is_leaf=True)  # arg181_1
    buf181 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf181, (1024, 256, 1, 1), is_leaf=True)  # arg182_1
    buf182 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf182, (1024,), is_leaf=True)  # arg183_1
    buf183 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf183, (1024,), is_leaf=True)  # arg184_1
    buf184 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf184, (1024,), is_leaf=True)  # arg185_1
    buf185 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf185, (1024,), is_leaf=True)  # arg186_1
    buf186 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf186, (256, 1024, 1, 1), is_leaf=True)  # arg187_1
    buf187 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf187, (256,), is_leaf=True)  # arg188_1
    buf188 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf188, (256,), is_leaf=True)  # arg189_1
    buf189 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf189, (256,), is_leaf=True)  # arg190_1
    buf190 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf190, (256,), is_leaf=True)  # arg191_1
    buf191 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf191, (256, 256, 3, 3), is_leaf=True)  # arg192_1
    buf192 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf192, (256,), is_leaf=True)  # arg193_1
    buf193 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf193, (256,), is_leaf=True)  # arg194_1
    buf194 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf194, (256,), is_leaf=True)  # arg195_1
    buf195 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf195, (256,), is_leaf=True)  # arg196_1
    buf196 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1024, 256, 1, 1), is_leaf=True)  # arg197_1
    buf197 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf197, (1024,), is_leaf=True)  # arg198_1
    buf198 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf198, (1024,), is_leaf=True)  # arg199_1
    buf199 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf199, (1024,), is_leaf=True)  # arg200_1
    buf200 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf200, (1024,), is_leaf=True)  # arg201_1
    buf201 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf201, (256, 1024, 1, 1), is_leaf=True)  # arg202_1
    buf202 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf202, (256,), is_leaf=True)  # arg203_1
    buf203 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf203, (256,), is_leaf=True)  # arg204_1
    buf204 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf204, (256,), is_leaf=True)  # arg205_1
    buf205 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf205, (256,), is_leaf=True)  # arg206_1
    buf206 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf206, (256, 256, 3, 3), is_leaf=True)  # arg207_1
    buf207 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf207, (256,), is_leaf=True)  # arg208_1
    buf208 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf208, (256,), is_leaf=True)  # arg209_1
    buf209 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf209, (256,), is_leaf=True)  # arg210_1
    buf210 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf210, (256,), is_leaf=True)  # arg211_1
    buf211 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf211, (1024, 256, 1, 1), is_leaf=True)  # arg212_1
    buf212 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf212, (1024,), is_leaf=True)  # arg213_1
    buf213 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1024,), is_leaf=True)  # arg214_1
    buf214 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf214, (1024,), is_leaf=True)  # arg215_1
    buf215 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1024,), is_leaf=True)  # arg216_1
    buf216 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf216, (512, 1024, 1, 1), is_leaf=True)  # arg217_1
    buf217 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf217, (512,), is_leaf=True)  # arg218_1
    buf218 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf218, (512,), is_leaf=True)  # arg219_1
    buf219 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf219, (512,), is_leaf=True)  # arg220_1
    buf220 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf220, (512,), is_leaf=True)  # arg221_1
    buf221 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf221, (512, 512, 3, 3), is_leaf=True)  # arg222_1
    buf222 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf222, (512,), is_leaf=True)  # arg223_1
    buf223 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf223, (512,), is_leaf=True)  # arg224_1
    buf224 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf224, (512,), is_leaf=True)  # arg225_1
    buf225 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf225, (512,), is_leaf=True)  # arg226_1
    buf226 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf226, (2048, 512, 1, 1), is_leaf=True)  # arg227_1
    buf227 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf227, (2048,), is_leaf=True)  # arg228_1
    buf228 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf228, (2048,), is_leaf=True)  # arg229_1
    buf229 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf229, (2048,), is_leaf=True)  # arg230_1
    buf230 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf230, (2048,), is_leaf=True)  # arg231_1
    buf231 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf231, (2048, 1024, 1, 1), is_leaf=True)  # arg232_1
    buf232 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf232, (2048,), is_leaf=True)  # arg233_1
    buf233 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf233, (2048,), is_leaf=True)  # arg234_1
    buf234 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf234, (2048,), is_leaf=True)  # arg235_1
    buf235 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf235, (2048,), is_leaf=True)  # arg236_1
    buf236 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf236, (512, 2048, 1, 1), is_leaf=True)  # arg237_1
    buf237 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf237, (512,), is_leaf=True)  # arg238_1
    buf238 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf238, (512,), is_leaf=True)  # arg239_1
    buf239 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf239, (512,), is_leaf=True)  # arg240_1
    buf240 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf240, (512,), is_leaf=True)  # arg241_1
    buf241 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf241, (512, 512, 3, 3), is_leaf=True)  # arg242_1
    buf242 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf242, (512,), is_leaf=True)  # arg243_1
    buf243 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf243, (512,), is_leaf=True)  # arg244_1
    buf244 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf244, (512,), is_leaf=True)  # arg245_1
    buf245 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf245, (512,), is_leaf=True)  # arg246_1
    buf246 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf246, (2048, 512, 1, 1), is_leaf=True)  # arg247_1
    buf247 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf247, (2048,), is_leaf=True)  # arg248_1
    buf248 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf248, (2048,), is_leaf=True)  # arg249_1
    buf249 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf249, (2048,), is_leaf=True)  # arg250_1
    buf250 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf250, (2048,), is_leaf=True)  # arg251_1
    buf251 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf251, (512, 2048, 1, 1), is_leaf=True)  # arg252_1
    buf252 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf252, (512,), is_leaf=True)  # arg253_1
    buf253 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf253, (512,), is_leaf=True)  # arg254_1
    buf254 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf254, (512,), is_leaf=True)  # arg255_1
    buf255 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf255, (512,), is_leaf=True)  # arg256_1
    buf256 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf256, (512, 512, 3, 3), is_leaf=True)  # arg257_1
    buf257 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf257, (512,), is_leaf=True)  # arg258_1
    buf258 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf258, (512,), is_leaf=True)  # arg259_1
    buf259 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf259, (512,), is_leaf=True)  # arg260_1
    buf260 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf260, (512,), is_leaf=True)  # arg261_1
    buf261 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf261, (2048, 512, 1, 1), is_leaf=True)  # arg262_1
    buf262 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf262, (2048,), is_leaf=True)  # arg263_1
    buf263 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf263, (2048,), is_leaf=True)  # arg264_1
    buf264 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf264, (2048,), is_leaf=True)  # arg265_1
    buf265 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf265, (2048,), is_leaf=True)  # arg266_1
    buf266 = reader.storage(None, 819200, device=device(type='cuda', index=0))
    reader.tensor(buf266, (100, 2048), is_leaf=True)  # arg267_1
    buf267 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf267, (100,), is_leaf=True)  # arg268_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)