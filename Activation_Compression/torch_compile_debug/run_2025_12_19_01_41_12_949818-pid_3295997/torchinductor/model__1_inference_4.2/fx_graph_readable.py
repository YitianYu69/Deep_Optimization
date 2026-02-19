class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 3, 7, 7]", arg1_1: "Sym(s77)", arg2_1: "f32[s77, 3, 224, 224]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg7_1: "f32[64, 64, 1, 1]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]", arg12_1: "f32[64, 64, 3, 3]", arg13_1: "f32[64]", arg14_1: "f32[64]", arg15_1: "f32[64]", arg16_1: "f32[64]", arg17_1: "f32[256, 64, 1, 1]", arg18_1: "f32[256]", arg19_1: "f32[256]", arg20_1: "f32[256]", arg21_1: "f32[256]", arg22_1: "f32[256, 64, 1, 1]", arg23_1: "f32[256]", arg24_1: "f32[256]", arg25_1: "f32[256]", arg26_1: "f32[256]", arg27_1: "f32[64, 256, 1, 1]", arg28_1: "f32[64]", arg29_1: "f32[64]", arg30_1: "f32[64]", arg31_1: "f32[64]", arg32_1: "f32[64, 64, 3, 3]", arg33_1: "f32[64]", arg34_1: "f32[64]", arg35_1: "f32[64]", arg36_1: "f32[64]", arg37_1: "f32[256, 64, 1, 1]", arg38_1: "f32[256]", arg39_1: "f32[256]", arg40_1: "f32[256]", arg41_1: "f32[256]", arg42_1: "f32[64, 256, 1, 1]", arg43_1: "f32[64]", arg44_1: "f32[64]", arg45_1: "f32[64]", arg46_1: "f32[64]", arg47_1: "f32[64, 64, 3, 3]", arg48_1: "f32[64]", arg49_1: "f32[64]", arg50_1: "f32[64]", arg51_1: "f32[64]", arg52_1: "f32[256, 64, 1, 1]", arg53_1: "f32[256]", arg54_1: "f32[256]", arg55_1: "f32[256]", arg56_1: "f32[256]", arg57_1: "f32[128, 256, 1, 1]", arg58_1: "f32[128]", arg59_1: "f32[128]", arg60_1: "f32[128]", arg61_1: "f32[128]", arg62_1: "f32[128, 128, 3, 3]", arg63_1: "f32[128]", arg64_1: "f32[128]", arg65_1: "f32[128]", arg66_1: "f32[128]", arg67_1: "f32[512, 128, 1, 1]", arg68_1: "f32[512]", arg69_1: "f32[512]", arg70_1: "f32[512]", arg71_1: "f32[512]", arg72_1: "f32[512, 256, 1, 1]", arg73_1: "f32[512]", arg74_1: "f32[512]", arg75_1: "f32[512]", arg76_1: "f32[512]", arg77_1: "f32[128, 512, 1, 1]", arg78_1: "f32[128]", arg79_1: "f32[128]", arg80_1: "f32[128]", arg81_1: "f32[128]", arg82_1: "f32[128, 128, 3, 3]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "f32[128]", arg86_1: "f32[128]", arg87_1: "f32[512, 128, 1, 1]", arg88_1: "f32[512]", arg89_1: "f32[512]", arg90_1: "f32[512]", arg91_1: "f32[512]", arg92_1: "f32[128, 512, 1, 1]", arg93_1: "f32[128]", arg94_1: "f32[128]", arg95_1: "f32[128]", arg96_1: "f32[128]", arg97_1: "f32[128, 128, 3, 3]", arg98_1: "f32[128]", arg99_1: "f32[128]", arg100_1: "f32[128]", arg101_1: "f32[128]", arg102_1: "f32[512, 128, 1, 1]", arg103_1: "f32[512]", arg104_1: "f32[512]", arg105_1: "f32[512]", arg106_1: "f32[512]", arg107_1: "f32[128, 512, 1, 1]", arg108_1: "f32[128]", arg109_1: "f32[128]", arg110_1: "f32[128]", arg111_1: "f32[128]", arg112_1: "f32[128, 128, 3, 3]", arg113_1: "f32[128]", arg114_1: "f32[128]", arg115_1: "f32[128]", arg116_1: "f32[128]", arg117_1: "f32[512, 128, 1, 1]", arg118_1: "f32[512]", arg119_1: "f32[512]", arg120_1: "f32[512]", arg121_1: "f32[512]", arg122_1: "f32[256, 512, 1, 1]", arg123_1: "f32[256]", arg124_1: "f32[256]", arg125_1: "f32[256]", arg126_1: "f32[256]", arg127_1: "f32[256, 256, 3, 3]", arg128_1: "f32[256]", arg129_1: "f32[256]", arg130_1: "f32[256]", arg131_1: "f32[256]", arg132_1: "f32[1024, 256, 1, 1]", arg133_1: "f32[1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 512, 1, 1]", arg138_1: "f32[1024]", arg139_1: "f32[1024]", arg140_1: "f32[1024]", arg141_1: "f32[1024]", arg142_1: "f32[256, 1024, 1, 1]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "f32[256]", arg146_1: "f32[256]", arg147_1: "f32[256, 256, 3, 3]", arg148_1: "f32[256]", arg149_1: "f32[256]", arg150_1: "f32[256]", arg151_1: "f32[256]", arg152_1: "f32[1024, 256, 1, 1]", arg153_1: "f32[1024]", arg154_1: "f32[1024]", arg155_1: "f32[1024]", arg156_1: "f32[1024]", arg157_1: "f32[256, 1024, 1, 1]", arg158_1: "f32[256]", arg159_1: "f32[256]", arg160_1: "f32[256]", arg161_1: "f32[256]", arg162_1: "f32[256, 256, 3, 3]", arg163_1: "f32[256]", arg164_1: "f32[256]", arg165_1: "f32[256]", arg166_1: "f32[256]", arg167_1: "f32[1024, 256, 1, 1]", arg168_1: "f32[1024]", arg169_1: "f32[1024]", arg170_1: "f32[1024]", arg171_1: "f32[1024]", arg172_1: "f32[256, 1024, 1, 1]", arg173_1: "f32[256]", arg174_1: "f32[256]", arg175_1: "f32[256]", arg176_1: "f32[256]", arg177_1: "f32[256, 256, 3, 3]", arg178_1: "f32[256]", arg179_1: "f32[256]", arg180_1: "f32[256]", arg181_1: "f32[256]", arg182_1: "f32[1024, 256, 1, 1]", arg183_1: "f32[1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024]", arg186_1: "f32[1024]", arg187_1: "f32[256, 1024, 1, 1]", arg188_1: "f32[256]", arg189_1: "f32[256]", arg190_1: "f32[256]", arg191_1: "f32[256]", arg192_1: "f32[256, 256, 3, 3]", arg193_1: "f32[256]", arg194_1: "f32[256]", arg195_1: "f32[256]", arg196_1: "f32[256]", arg197_1: "f32[1024, 256, 1, 1]", arg198_1: "f32[1024]", arg199_1: "f32[1024]", arg200_1: "f32[1024]", arg201_1: "f32[1024]", arg202_1: "f32[256, 1024, 1, 1]", arg203_1: "f32[256]", arg204_1: "f32[256]", arg205_1: "f32[256]", arg206_1: "f32[256]", arg207_1: "f32[256, 256, 3, 3]", arg208_1: "f32[256]", arg209_1: "f32[256]", arg210_1: "f32[256]", arg211_1: "f32[256]", arg212_1: "f32[1024, 256, 1, 1]", arg213_1: "f32[1024]", arg214_1: "f32[1024]", arg215_1: "f32[1024]", arg216_1: "f32[1024]", arg217_1: "f32[512, 1024, 1, 1]", arg218_1: "f32[512]", arg219_1: "f32[512]", arg220_1: "f32[512]", arg221_1: "f32[512]", arg222_1: "f32[512, 512, 3, 3]", arg223_1: "f32[512]", arg224_1: "f32[512]", arg225_1: "f32[512]", arg226_1: "f32[512]", arg227_1: "f32[2048, 512, 1, 1]", arg228_1: "f32[2048]", arg229_1: "f32[2048]", arg230_1: "f32[2048]", arg231_1: "f32[2048]", arg232_1: "f32[2048, 1024, 1, 1]", arg233_1: "f32[2048]", arg234_1: "f32[2048]", arg235_1: "f32[2048]", arg236_1: "f32[2048]", arg237_1: "f32[512, 2048, 1, 1]", arg238_1: "f32[512]", arg239_1: "f32[512]", arg240_1: "f32[512]", arg241_1: "f32[512]", arg242_1: "f32[512, 512, 3, 3]", arg243_1: "f32[512]", arg244_1: "f32[512]", arg245_1: "f32[512]", arg246_1: "f32[512]", arg247_1: "f32[2048, 512, 1, 1]", arg248_1: "f32[2048]", arg249_1: "f32[2048]", arg250_1: "f32[2048]", arg251_1: "f32[2048]", arg252_1: "f32[512, 2048, 1, 1]", arg253_1: "f32[512]", arg254_1: "f32[512]", arg255_1: "f32[512]", arg256_1: "f32[512]", arg257_1: "f32[512, 512, 3, 3]", arg258_1: "f32[512]", arg259_1: "f32[512]", arg260_1: "f32[512]", arg261_1: "f32[512]", arg262_1: "f32[2048, 512, 1, 1]", arg263_1: "f32[2048]", arg264_1: "f32[2048]", arg265_1: "f32[2048]", arg266_1: "f32[2048]", arg267_1: "f32[100, 2048]", arg268_1: "f32[100]"):
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type: "bf16[64, 3, 7, 7]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[s77, 3, 224, 224]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        convolution: "bf16[s77, 64, 112, 112]" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  convert_element_type_1 = convert_element_type = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_10: "f32[64]" = torch.ops.aten.add.Tensor(arg4_1, 1e-05);  arg4_1 = None
        sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
        reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul_7: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_7, -1);  mul_7 = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        sub_2: "f32[s77, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        mul_8: "f32[s77, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_3);  sub_2 = unsqueeze_3 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_9: "f32[s77, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_11: "f32[s77, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_9, unsqueeze_7);  mul_9 = unsqueeze_7 = None
        convert_element_type_4: "bf16[s77, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu: "bf16[s77, 64, 112, 112]" = torch.ops.aten.relu.default(convert_element_type_4);  convert_element_type_4 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:8 in forward, code: maxpool = self.maxpool(relu);  relu = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem: "bf16[s77, 64, 56, 56]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_5: "bf16[64, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convolution_1: "bf16[s77, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem, convert_element_type_5, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_5 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_37: "f32[64]" = torch.ops.aten.add.Tensor(arg9_1, 1e-05);  arg9_1 = None
        sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
        reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_23: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_23, -1);  mul_23 = None
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        sub_8: "f32[s77, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        mul_24: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_11);  sub_8 = unsqueeze_11 = None
        unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_25: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_24, unsqueeze_13);  mul_24 = unsqueeze_13 = None
        unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_38: "f32[s77, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_15);  mul_25 = unsqueeze_15 = None
        convert_element_type_8: "bf16[s77, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_1: "bf16[s77, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_8);  convert_element_type_8 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_9: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        convolution_2: "bf16[s77, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, convert_element_type_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = convert_element_type_9 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_54: "f32[64]" = torch.ops.aten.add.Tensor(arg14_1, 1e-05);  arg14_1 = None
        sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_35: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_35, -1);  mul_35 = None
        unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        sub_12: "f32[s77, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        mul_36: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_19);  sub_12 = unsqueeze_19 = None
        unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_37: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_21);  mul_36 = unsqueeze_21 = None
        unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_55: "f32[s77, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_23);  mul_37 = unsqueeze_23 = None
        convert_element_type_12: "bf16[s77, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_2: "bf16[s77, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_12);  convert_element_type_12 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_13: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        convolution_3: "bf16[s77, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_2, convert_element_type_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = convert_element_type_13 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_71: "f32[256]" = torch.ops.aten.add.Tensor(arg19_1, 1e-05);  arg19_1 = None
        sqrt_3: "f32[256]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
        reciprocal_3: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_47: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_47, -1);  mul_47 = None
        unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        sub_16: "f32[s77, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        mul_48: "f32[s77, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_27);  sub_16 = unsqueeze_27 = None
        unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_49: "f32[s77, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_48, unsqueeze_29);  mul_48 = unsqueeze_29 = None
        unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_72: "f32[s77, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_31);  mul_49 = unsqueeze_31 = None
        convert_element_type_16: "bf16[s77, 256, 56, 56]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_17: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        convolution_4: "bf16[s77, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem, convert_element_type_17, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = convert_element_type_17 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_83: "f32[256]" = torch.ops.aten.add.Tensor(arg24_1, 1e-05);  arg24_1 = None
        sqrt_4: "f32[256]" = torch.ops.aten.sqrt.default(add_83);  add_83 = None
        reciprocal_4: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_57: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
        unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        sub_19: "f32[s77, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        mul_58: "f32[s77, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_35);  sub_19 = unsqueeze_35 = None
        unsqueeze_36: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_37: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_59: "f32[s77, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_37);  mul_58 = unsqueeze_37 = None
        unsqueeze_38: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_39: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_84: "f32[s77, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_39);  mul_59 = unsqueeze_39 = None
        convert_element_type_20: "bf16[s77, 256, 56, 56]" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:19 in forward, code: add = layer1_0_bn3 + layer1_0_downsample_1;  layer1_0_bn3 = layer1_0_downsample_1 = None
        add_90: "bf16[s77, 256, 56, 56]" = torch.ops.aten.add.Tensor(convert_element_type_16, convert_element_type_20);  convert_element_type_16 = convert_element_type_20 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_3: "bf16[s77, 256, 56, 56]" = torch.ops.aten.relu.default(add_90);  add_90 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_21: "bf16[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        convolution_5: "bf16[s77, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_3, convert_element_type_21, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_21 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_106: "f32[64]" = torch.ops.aten.add.Tensor(arg29_1, 1e-05);  arg29_1 = None
        sqrt_5: "f32[64]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_5: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_71: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_71, -1);  mul_71 = None
        unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        sub_24: "f32[s77, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        mul_72: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_43);  sub_24 = unsqueeze_43 = None
        unsqueeze_44: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_45: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_73: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_72, unsqueeze_45);  mul_72 = unsqueeze_45 = None
        unsqueeze_46: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_47: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_107: "f32[s77, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_73, unsqueeze_47);  mul_73 = unsqueeze_47 = None
        convert_element_type_24: "bf16[s77, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16);  add_107 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_4: "bf16[s77, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_24);  convert_element_type_24 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_25: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        convolution_6: "bf16[s77, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_4, convert_element_type_25, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = convert_element_type_25 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_123: "f32[64]" = torch.ops.aten.add.Tensor(arg34_1, 1e-05);  arg34_1 = None
        sqrt_6: "f32[64]" = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_6: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_83: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_83, -1);  mul_83 = None
        unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        sub_28: "f32[s77, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        mul_84: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_51);  sub_28 = unsqueeze_51 = None
        unsqueeze_52: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_53: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_85: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_53);  mul_84 = unsqueeze_53 = None
        unsqueeze_54: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_55: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_124: "f32[s77, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_55);  mul_85 = unsqueeze_55 = None
        convert_element_type_28: "bf16[s77, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_5: "bf16[s77, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_28);  convert_element_type_28 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_29: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        convolution_7: "bf16[s77, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, convert_element_type_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = convert_element_type_29 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_140: "f32[256]" = torch.ops.aten.add.Tensor(arg39_1, 1e-05);  arg39_1 = None
        sqrt_7: "f32[256]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_7: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_95: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_95, -1);  mul_95 = None
        unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        sub_32: "f32[s77, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        mul_96: "f32[s77, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_59);  sub_32 = unsqueeze_59 = None
        unsqueeze_60: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_61: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_97: "f32[s77, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_61);  mul_96 = unsqueeze_61 = None
        unsqueeze_62: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_63: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_141: "f32[s77, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_63);  mul_97 = unsqueeze_63 = None
        convert_element_type_32: "bf16[s77, 256, 56, 56]" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16);  add_141 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:29 in forward, code: add_1 = layer1_1_bn3 + layer1_0_relu_2;  layer1_1_bn3 = layer1_0_relu_2 = None
        add_147: "bf16[s77, 256, 56, 56]" = torch.ops.aten.add.Tensor(convert_element_type_32, relu_3);  convert_element_type_32 = relu_3 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_6: "bf16[s77, 256, 56, 56]" = torch.ops.aten.relu.default(add_147);  add_147 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_33: "bf16[64, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        convolution_8: "bf16[s77, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_6, convert_element_type_33, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_33 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_163: "f32[64]" = torch.ops.aten.add.Tensor(arg44_1, 1e-05);  arg44_1 = None
        sqrt_8: "f32[64]" = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_8: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_109: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_64: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_65: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        unsqueeze_66: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_109, -1);  mul_109 = None
        unsqueeze_67: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        sub_37: "f32[s77, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        mul_110: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_67);  sub_37 = unsqueeze_67 = None
        unsqueeze_68: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_69: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_111: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_110, unsqueeze_69);  mul_110 = unsqueeze_69 = None
        unsqueeze_70: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_71: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_164: "f32[s77, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_71);  mul_111 = unsqueeze_71 = None
        convert_element_type_36: "bf16[s77, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_7: "bf16[s77, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_36);  convert_element_type_36 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_37: "bf16[64, 64, 3, 3]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        convolution_9: "bf16[s77, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_7, convert_element_type_37, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = convert_element_type_37 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_180: "f32[64]" = torch.ops.aten.add.Tensor(arg49_1, 1e-05);  arg49_1 = None
        sqrt_9: "f32[64]" = torch.ops.aten.sqrt.default(add_180);  add_180 = None
        reciprocal_9: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_121: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_121, -1);  mul_121 = None
        unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        sub_41: "f32[s77, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        mul_122: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_75);  sub_41 = unsqueeze_75 = None
        unsqueeze_76: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_77: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_123: "f32[s77, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_122, unsqueeze_77);  mul_122 = unsqueeze_77 = None
        unsqueeze_78: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_79: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_181: "f32[s77, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_123, unsqueeze_79);  mul_123 = unsqueeze_79 = None
        convert_element_type_40: "bf16[s77, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_181, torch.bfloat16);  add_181 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_8: "bf16[s77, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_40);  convert_element_type_40 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_41: "bf16[256, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        convolution_10: "bf16[s77, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_8, convert_element_type_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = convert_element_type_41 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_197: "f32[256]" = torch.ops.aten.add.Tensor(arg54_1, 1e-05);  arg54_1 = None
        sqrt_10: "f32[256]" = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_10: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_133: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, -1);  arg53_1 = None
        unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_133, -1);  mul_133 = None
        unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        sub_45: "f32[s77, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        mul_134: "f32[s77, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_83);  sub_45 = unsqueeze_83 = None
        unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_135: "f32[s77, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_134, unsqueeze_85);  mul_134 = unsqueeze_85 = None
        unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_198: "f32[s77, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_87);  mul_135 = unsqueeze_87 = None
        convert_element_type_44: "bf16[s77, 256, 56, 56]" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16);  add_198 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:39 in forward, code: add_2 = layer1_2_bn3 + layer1_1_relu_2;  layer1_2_bn3 = layer1_1_relu_2 = None
        add_204: "bf16[s77, 256, 56, 56]" = torch.ops.aten.add.Tensor(convert_element_type_44, relu_6);  convert_element_type_44 = relu_6 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_9: "bf16[s77, 256, 56, 56]" = torch.ops.aten.relu.default(add_204);  add_204 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_45: "bf16[128, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        convolution_11: "bf16[s77, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_9, convert_element_type_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_45 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_220: "f32[128]" = torch.ops.aten.add.Tensor(arg59_1, 1e-05);  arg59_1 = None
        sqrt_11: "f32[128]" = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_11: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_147: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        sub_50: "f32[s77, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
        mul_148: "f32[s77, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_91);  sub_50 = unsqueeze_91 = None
        unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_149: "f32[s77, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_93);  mul_148 = unsqueeze_93 = None
        unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_221: "f32[s77, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_95);  mul_149 = unsqueeze_95 = None
        convert_element_type_48: "bf16[s77, 128, 56, 56]" = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_10: "bf16[s77, 128, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_48);  convert_element_type_48 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_49: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        convolution_12: "bf16[s77, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_10, convert_element_type_49, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = convert_element_type_49 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_237: "f32[128]" = torch.ops.aten.add.Tensor(arg64_1, 1e-05);  arg64_1 = None
        sqrt_12: "f32[128]" = torch.ops.aten.sqrt.default(add_237);  add_237 = None
        reciprocal_12: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_159: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        sub_54: "f32[s77, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
        mul_160: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_99);  sub_54 = unsqueeze_99 = None
        unsqueeze_100: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_101: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_161: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_101);  mul_160 = unsqueeze_101 = None
        unsqueeze_102: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_103: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_238: "f32[s77, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_103);  mul_161 = unsqueeze_103 = None
        convert_element_type_52: "bf16[s77, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_238, torch.bfloat16);  add_238 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_11: "bf16[s77, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_52);  convert_element_type_52 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_53: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convolution_13: "bf16[s77, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_11, convert_element_type_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = convert_element_type_53 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_254: "f32[512]" = torch.ops.aten.add.Tensor(arg69_1, 1e-05);  arg69_1 = None
        sqrt_13: "f32[512]" = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_13: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_171: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_104: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_105: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        unsqueeze_106: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_107: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        sub_58: "f32[s77, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
        mul_172: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_107);  sub_58 = unsqueeze_107 = None
        unsqueeze_108: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_109: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_173: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_109);  mul_172 = unsqueeze_109 = None
        unsqueeze_110: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_111: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_255: "f32[s77, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_111);  mul_173 = unsqueeze_111 = None
        convert_element_type_56: "bf16[s77, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_255, torch.bfloat16);  add_255 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_57: "bf16[512, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        convolution_14: "bf16[s77, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_9, convert_element_type_57, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = convert_element_type_57 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_266: "f32[512]" = torch.ops.aten.add.Tensor(arg74_1, 1e-05);  arg74_1 = None
        sqrt_14: "f32[512]" = torch.ops.aten.sqrt.default(add_266);  add_266 = None
        reciprocal_14: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_181: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_112: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_113: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        unsqueeze_114: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_181, -1);  mul_181 = None
        unsqueeze_115: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        sub_61: "f32[s77, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
        mul_182: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_115);  sub_61 = unsqueeze_115 = None
        unsqueeze_116: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_117: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_183: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_117);  mul_182 = unsqueeze_117 = None
        unsqueeze_118: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_119: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_267: "f32[s77, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_183, unsqueeze_119);  mul_183 = unsqueeze_119 = None
        convert_element_type_60: "bf16[s77, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_267, torch.bfloat16);  add_267 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:51 in forward, code: add_3 = layer2_0_bn3 + layer2_0_downsample_1;  layer2_0_bn3 = layer2_0_downsample_1 = None
        add_273: "bf16[s77, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_56, convert_element_type_60);  convert_element_type_56 = convert_element_type_60 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_12: "bf16[s77, 512, 28, 28]" = torch.ops.aten.relu.default(add_273);  add_273 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_61: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        convolution_15: "bf16[s77, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_12, convert_element_type_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_61 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_289: "f32[128]" = torch.ops.aten.add.Tensor(arg79_1, 1e-05);  arg79_1 = None
        sqrt_15: "f32[128]" = torch.ops.aten.sqrt.default(add_289);  add_289 = None
        reciprocal_15: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_195: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        sub_66: "f32[s77, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
        mul_196: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_123);  sub_66 = unsqueeze_123 = None
        unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_197: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_125);  mul_196 = unsqueeze_125 = None
        unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_290: "f32[s77, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_127);  mul_197 = unsqueeze_127 = None
        convert_element_type_64: "bf16[s77, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_290, torch.bfloat16);  add_290 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_13: "bf16[s77, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_64);  convert_element_type_64 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_65: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(arg82_1, torch.bfloat16);  arg82_1 = None
        convolution_16: "bf16[s77, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_13, convert_element_type_65, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = convert_element_type_65 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_306: "f32[128]" = torch.ops.aten.add.Tensor(arg84_1, 1e-05);  arg84_1 = None
        sqrt_16: "f32[128]" = torch.ops.aten.sqrt.default(add_306);  add_306 = None
        reciprocal_16: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_207: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_128: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_129: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        unsqueeze_130: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_131: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        sub_70: "f32[s77, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
        mul_208: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_131);  sub_70 = unsqueeze_131 = None
        unsqueeze_132: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_133: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_209: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_133);  mul_208 = unsqueeze_133 = None
        unsqueeze_134: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_135: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_307: "f32[s77, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_135);  mul_209 = unsqueeze_135 = None
        convert_element_type_68: "bf16[s77, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_307, torch.bfloat16);  add_307 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_14: "bf16[s77, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_68);  convert_element_type_68 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_69: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.bfloat16);  arg87_1 = None
        convolution_17: "bf16[s77, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_14, convert_element_type_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = convert_element_type_69 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_323: "f32[512]" = torch.ops.aten.add.Tensor(arg89_1, 1e-05);  arg89_1 = None
        sqrt_17: "f32[512]" = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_17: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_219: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_136: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_137: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        unsqueeze_138: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_139: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        sub_74: "f32[s77, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
        mul_220: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_139);  sub_74 = unsqueeze_139 = None
        unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_221: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_141);  mul_220 = unsqueeze_141 = None
        unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_324: "f32[s77, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_143);  mul_221 = unsqueeze_143 = None
        convert_element_type_72: "bf16[s77, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_324, torch.bfloat16);  add_324 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:61 in forward, code: add_4 = layer2_1_bn3 + layer2_0_relu_2;  layer2_1_bn3 = layer2_0_relu_2 = None
        add_330: "bf16[s77, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_72, relu_12);  convert_element_type_72 = relu_12 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_15: "bf16[s77, 512, 28, 28]" = torch.ops.aten.relu.default(add_330);  add_330 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_73: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.bfloat16);  arg92_1 = None
        convolution_18: "bf16[s77, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_15, convert_element_type_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_73 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_346: "f32[128]" = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
        sqrt_18: "f32[128]" = torch.ops.aten.sqrt.default(add_346);  add_346 = None
        reciprocal_18: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_233: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_144: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_145: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        unsqueeze_146: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_233, -1);  mul_233 = None
        unsqueeze_147: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        sub_79: "f32[s77, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
        mul_234: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_147);  sub_79 = unsqueeze_147 = None
        unsqueeze_148: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_149: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_235: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_149);  mul_234 = unsqueeze_149 = None
        unsqueeze_150: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_151: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_347: "f32[s77, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_235, unsqueeze_151);  mul_235 = unsqueeze_151 = None
        convert_element_type_76: "bf16[s77, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_347, torch.bfloat16);  add_347 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_16: "bf16[s77, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_76);  convert_element_type_76 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_77: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(arg97_1, torch.bfloat16);  arg97_1 = None
        convolution_19: "bf16[s77, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_16, convert_element_type_77, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = convert_element_type_77 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_363: "f32[128]" = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
        sqrt_19: "f32[128]" = torch.ops.aten.sqrt.default(add_363);  add_363 = None
        reciprocal_19: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_245: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_152: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_153: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        unsqueeze_154: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_245, -1);  mul_245 = None
        unsqueeze_155: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        sub_83: "f32[s77, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
        mul_246: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_155);  sub_83 = unsqueeze_155 = None
        unsqueeze_156: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_157: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_247: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_157);  mul_246 = unsqueeze_157 = None
        unsqueeze_158: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_159: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_364: "f32[s77, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_247, unsqueeze_159);  mul_247 = unsqueeze_159 = None
        convert_element_type_80: "bf16[s77, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_364, torch.bfloat16);  add_364 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_17: "bf16[s77, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_80);  convert_element_type_80 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_81: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        convolution_20: "bf16[s77, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_17, convert_element_type_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = convert_element_type_81 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_380: "f32[512]" = torch.ops.aten.add.Tensor(arg104_1, 1e-05);  arg104_1 = None
        sqrt_20: "f32[512]" = torch.ops.aten.sqrt.default(add_380);  add_380 = None
        reciprocal_20: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_257: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        unsqueeze_160: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_161: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        unsqueeze_162: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_257, -1);  mul_257 = None
        unsqueeze_163: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        sub_87: "f32[s77, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  convolution_20 = unsqueeze_161 = None
        mul_258: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_163);  sub_87 = unsqueeze_163 = None
        unsqueeze_164: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_165: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_259: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_165);  mul_258 = unsqueeze_165 = None
        unsqueeze_166: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_167: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_381: "f32[s77, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_259, unsqueeze_167);  mul_259 = unsqueeze_167 = None
        convert_element_type_84: "bf16[s77, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_381, torch.bfloat16);  add_381 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:71 in forward, code: add_5 = layer2_2_bn3 + layer2_1_relu_2;  layer2_2_bn3 = layer2_1_relu_2 = None
        add_387: "bf16[s77, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_84, relu_15);  convert_element_type_84 = relu_15 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_18: "bf16[s77, 512, 28, 28]" = torch.ops.aten.relu.default(add_387);  add_387 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_85: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        convolution_21: "bf16[s77, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_18, convert_element_type_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_85 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_403: "f32[128]" = torch.ops.aten.add.Tensor(arg109_1, 1e-05);  arg109_1 = None
        sqrt_21: "f32[128]" = torch.ops.aten.sqrt.default(add_403);  add_403 = None
        reciprocal_21: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_271: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        unsqueeze_168: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_169: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        unsqueeze_170: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_271, -1);  mul_271 = None
        unsqueeze_171: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        sub_92: "f32[s77, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  convolution_21 = unsqueeze_169 = None
        mul_272: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_171);  sub_92 = unsqueeze_171 = None
        unsqueeze_172: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_173: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_273: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_272, unsqueeze_173);  mul_272 = unsqueeze_173 = None
        unsqueeze_174: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_175: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_404: "f32[s77, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_273, unsqueeze_175);  mul_273 = unsqueeze_175 = None
        convert_element_type_88: "bf16[s77, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_404, torch.bfloat16);  add_404 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_19: "bf16[s77, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_88);  convert_element_type_88 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_89: "bf16[128, 128, 3, 3]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        convolution_22: "bf16[s77, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_19, convert_element_type_89, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_19 = convert_element_type_89 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_420: "f32[128]" = torch.ops.aten.add.Tensor(arg114_1, 1e-05);  arg114_1 = None
        sqrt_22: "f32[128]" = torch.ops.aten.sqrt.default(add_420);  add_420 = None
        reciprocal_22: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_283: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        unsqueeze_176: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_177: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        unsqueeze_178: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
        unsqueeze_179: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        sub_96: "f32[s77, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  convolution_22 = unsqueeze_177 = None
        mul_284: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_179);  sub_96 = unsqueeze_179 = None
        unsqueeze_180: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_181: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_285: "f32[s77, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_181);  mul_284 = unsqueeze_181 = None
        unsqueeze_182: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_183: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_421: "f32[s77, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_285, unsqueeze_183);  mul_285 = unsqueeze_183 = None
        convert_element_type_92: "bf16[s77, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_421, torch.bfloat16);  add_421 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_20: "bf16[s77, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_92);  convert_element_type_92 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_93: "bf16[512, 128, 1, 1]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        convolution_23: "bf16[s77, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_20, convert_element_type_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = convert_element_type_93 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_437: "f32[512]" = torch.ops.aten.add.Tensor(arg119_1, 1e-05);  arg119_1 = None
        sqrt_23: "f32[512]" = torch.ops.aten.sqrt.default(add_437);  add_437 = None
        reciprocal_23: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_295: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_184: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_185: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        unsqueeze_186: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_187: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        sub_100: "f32[s77, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  convolution_23 = unsqueeze_185 = None
        mul_296: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_187);  sub_100 = unsqueeze_187 = None
        unsqueeze_188: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_189: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_297: "f32[s77, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_189);  mul_296 = unsqueeze_189 = None
        unsqueeze_190: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_191: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_438: "f32[s77, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_297, unsqueeze_191);  mul_297 = unsqueeze_191 = None
        convert_element_type_96: "bf16[s77, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_438, torch.bfloat16);  add_438 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:81 in forward, code: add_6 = layer2_3_bn3 + layer2_2_relu_2;  layer2_3_bn3 = layer2_2_relu_2 = None
        add_444: "bf16[s77, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_96, relu_18);  convert_element_type_96 = relu_18 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_21: "bf16[s77, 512, 28, 28]" = torch.ops.aten.relu.default(add_444);  add_444 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_97: "bf16[256, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        convolution_24: "bf16[s77, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_21, convert_element_type_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_97 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_460: "f32[256]" = torch.ops.aten.add.Tensor(arg124_1, 1e-05);  arg124_1 = None
        sqrt_24: "f32[256]" = torch.ops.aten.sqrt.default(add_460);  add_460 = None
        reciprocal_24: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_309: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_192: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_193: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        unsqueeze_194: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_195: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        sub_105: "f32[s77, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  convolution_24 = unsqueeze_193 = None
        mul_310: "f32[s77, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_195);  sub_105 = unsqueeze_195 = None
        unsqueeze_196: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_197: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_311: "f32[s77, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_197);  mul_310 = unsqueeze_197 = None
        unsqueeze_198: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_199: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_461: "f32[s77, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_199);  mul_311 = unsqueeze_199 = None
        convert_element_type_100: "bf16[s77, 256, 28, 28]" = torch.ops.prims.convert_element_type.default(add_461, torch.bfloat16);  add_461 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_22: "bf16[s77, 256, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_100);  convert_element_type_100 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_101: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        convolution_25: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_22, convert_element_type_101, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_22 = convert_element_type_101 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_477: "f32[256]" = torch.ops.aten.add.Tensor(arg129_1, 1e-05);  arg129_1 = None
        sqrt_25: "f32[256]" = torch.ops.aten.sqrt.default(add_477);  add_477 = None
        reciprocal_25: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_321: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_200: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_201: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        unsqueeze_202: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
        unsqueeze_203: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        sub_109: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  convolution_25 = unsqueeze_201 = None
        mul_322: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_203);  sub_109 = unsqueeze_203 = None
        unsqueeze_204: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_205: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_323: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_205);  mul_322 = unsqueeze_205 = None
        unsqueeze_206: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_207: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_478: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_207);  mul_323 = unsqueeze_207 = None
        convert_element_type_104: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_478, torch.bfloat16);  add_478 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_23: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_104);  convert_element_type_104 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_105: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        convolution_26: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_23, convert_element_type_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = convert_element_type_105 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_494: "f32[1024]" = torch.ops.aten.add.Tensor(arg134_1, 1e-05);  arg134_1 = None
        sqrt_26: "f32[1024]" = torch.ops.aten.sqrt.default(add_494);  add_494 = None
        reciprocal_26: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_333: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_208: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_209: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        unsqueeze_210: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_211: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        sub_113: "f32[s77, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  convolution_26 = unsqueeze_209 = None
        mul_334: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_211);  sub_113 = unsqueeze_211 = None
        unsqueeze_212: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_213: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_335: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_213);  mul_334 = unsqueeze_213 = None
        unsqueeze_214: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_215: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_495: "f32[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_215);  mul_335 = unsqueeze_215 = None
        convert_element_type_108: "bf16[s77, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_495, torch.bfloat16);  add_495 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_109: "bf16[1024, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        convolution_27: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_21, convert_element_type_109, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = convert_element_type_109 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_506: "f32[1024]" = torch.ops.aten.add.Tensor(arg139_1, 1e-05);  arg139_1 = None
        sqrt_27: "f32[1024]" = torch.ops.aten.sqrt.default(add_506);  add_506 = None
        reciprocal_27: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_343: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_216: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_217: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        unsqueeze_218: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_343, -1);  mul_343 = None
        unsqueeze_219: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        sub_116: "f32[s77, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  convolution_27 = unsqueeze_217 = None
        mul_344: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_219);  sub_116 = unsqueeze_219 = None
        unsqueeze_220: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_221: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_345: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_344, unsqueeze_221);  mul_344 = unsqueeze_221 = None
        unsqueeze_222: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_223: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_507: "f32[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_345, unsqueeze_223);  mul_345 = unsqueeze_223 = None
        convert_element_type_112: "bf16[s77, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_507, torch.bfloat16);  add_507 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:93 in forward, code: add_7 = layer3_0_bn3 + layer3_0_downsample_1;  layer3_0_bn3 = layer3_0_downsample_1 = None
        add_513: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_108, convert_element_type_112);  convert_element_type_108 = convert_element_type_112 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_24: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.relu.default(add_513);  add_513 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_113: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.bfloat16);  arg142_1 = None
        convolution_28: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_24, convert_element_type_113, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_113 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_529: "f32[256]" = torch.ops.aten.add.Tensor(arg144_1, 1e-05);  arg144_1 = None
        sqrt_28: "f32[256]" = torch.ops.aten.sqrt.default(add_529);  add_529 = None
        reciprocal_28: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_357: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_224: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_225: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        unsqueeze_226: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
        unsqueeze_227: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        sub_121: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  convolution_28 = unsqueeze_225 = None
        mul_358: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_227);  sub_121 = unsqueeze_227 = None
        unsqueeze_228: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_229: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_359: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_229);  mul_358 = unsqueeze_229 = None
        unsqueeze_230: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_231: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_530: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_231);  mul_359 = unsqueeze_231 = None
        convert_element_type_116: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_530, torch.bfloat16);  add_530 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_25: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_116);  convert_element_type_116 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_117: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.bfloat16);  arg147_1 = None
        convolution_29: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_25, convert_element_type_117, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_25 = convert_element_type_117 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_546: "f32[256]" = torch.ops.aten.add.Tensor(arg149_1, 1e-05);  arg149_1 = None
        sqrt_29: "f32[256]" = torch.ops.aten.sqrt.default(add_546);  add_546 = None
        reciprocal_29: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_369: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_232: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_233: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        unsqueeze_234: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_235: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        sub_125: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  convolution_29 = unsqueeze_233 = None
        mul_370: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_235);  sub_125 = unsqueeze_235 = None
        unsqueeze_236: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_237: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_371: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_237);  mul_370 = unsqueeze_237 = None
        unsqueeze_238: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_239: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_547: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_239);  mul_371 = unsqueeze_239 = None
        convert_element_type_120: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_547, torch.bfloat16);  add_547 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_26: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_120);  convert_element_type_120 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_121: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.bfloat16);  arg152_1 = None
        convolution_30: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_26, convert_element_type_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_26 = convert_element_type_121 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_563: "f32[1024]" = torch.ops.aten.add.Tensor(arg154_1, 1e-05);  arg154_1 = None
        sqrt_30: "f32[1024]" = torch.ops.aten.sqrt.default(add_563);  add_563 = None
        reciprocal_30: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_381: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_240: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_241: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        unsqueeze_242: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_243: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        sub_129: "f32[s77, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  convolution_30 = unsqueeze_241 = None
        mul_382: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_243);  sub_129 = unsqueeze_243 = None
        unsqueeze_244: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_245: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_383: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_245);  mul_382 = unsqueeze_245 = None
        unsqueeze_246: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_247: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_564: "f32[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_383, unsqueeze_247);  mul_383 = unsqueeze_247 = None
        convert_element_type_124: "bf16[s77, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_564, torch.bfloat16);  add_564 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:103 in forward, code: add_8 = layer3_1_bn3 + layer3_0_relu_2;  layer3_1_bn3 = layer3_0_relu_2 = None
        add_570: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_124, relu_24);  convert_element_type_124 = relu_24 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_27: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.relu.default(add_570);  add_570 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_125: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        convolution_31: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_27, convert_element_type_125, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_125 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_586: "f32[256]" = torch.ops.aten.add.Tensor(arg159_1, 1e-05);  arg159_1 = None
        sqrt_31: "f32[256]" = torch.ops.aten.sqrt.default(add_586);  add_586 = None
        reciprocal_31: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_395: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_248: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_249: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        unsqueeze_250: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_395, -1);  mul_395 = None
        unsqueeze_251: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        sub_134: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  convolution_31 = unsqueeze_249 = None
        mul_396: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_251);  sub_134 = unsqueeze_251 = None
        unsqueeze_252: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_253: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_397: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_396, unsqueeze_253);  mul_396 = unsqueeze_253 = None
        unsqueeze_254: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_255: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_587: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_397, unsqueeze_255);  mul_397 = unsqueeze_255 = None
        convert_element_type_128: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_587, torch.bfloat16);  add_587 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_28: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_128);  convert_element_type_128 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_129: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.bfloat16);  arg162_1 = None
        convolution_32: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_28, convert_element_type_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_28 = convert_element_type_129 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_603: "f32[256]" = torch.ops.aten.add.Tensor(arg164_1, 1e-05);  arg164_1 = None
        sqrt_32: "f32[256]" = torch.ops.aten.sqrt.default(add_603);  add_603 = None
        reciprocal_32: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_407: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_256: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_257: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        unsqueeze_258: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_407, -1);  mul_407 = None
        unsqueeze_259: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        sub_138: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  convolution_32 = unsqueeze_257 = None
        mul_408: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_259);  sub_138 = unsqueeze_259 = None
        unsqueeze_260: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_261: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_409: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_261);  mul_408 = unsqueeze_261 = None
        unsqueeze_262: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_263: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_604: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_263);  mul_409 = unsqueeze_263 = None
        convert_element_type_132: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_604, torch.bfloat16);  add_604 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_29: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_132);  convert_element_type_132 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_133: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        convolution_33: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_29, convert_element_type_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_29 = convert_element_type_133 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_620: "f32[1024]" = torch.ops.aten.add.Tensor(arg169_1, 1e-05);  arg169_1 = None
        sqrt_33: "f32[1024]" = torch.ops.aten.sqrt.default(add_620);  add_620 = None
        reciprocal_33: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_419: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_264: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_265: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_419, -1);  mul_419 = None
        unsqueeze_267: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        sub_142: "f32[s77, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  convolution_33 = unsqueeze_265 = None
        mul_420: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_267);  sub_142 = unsqueeze_267 = None
        unsqueeze_268: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_269: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_421: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_269);  mul_420 = unsqueeze_269 = None
        unsqueeze_270: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, -1);  arg171_1 = None
        unsqueeze_271: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_621: "f32[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_421, unsqueeze_271);  mul_421 = unsqueeze_271 = None
        convert_element_type_136: "bf16[s77, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_621, torch.bfloat16);  add_621 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:113 in forward, code: add_9 = layer3_2_bn3 + layer3_1_relu_2;  layer3_2_bn3 = layer3_1_relu_2 = None
        add_627: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_136, relu_27);  convert_element_type_136 = relu_27 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_30: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.relu.default(add_627);  add_627 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_137: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.bfloat16);  arg172_1 = None
        convolution_34: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_30, convert_element_type_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_137 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_643: "f32[256]" = torch.ops.aten.add.Tensor(arg174_1, 1e-05);  arg174_1 = None
        sqrt_34: "f32[256]" = torch.ops.aten.sqrt.default(add_643);  add_643 = None
        reciprocal_34: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_433: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_272: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_273: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        unsqueeze_274: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_433, -1);  mul_433 = None
        unsqueeze_275: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        sub_147: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  convolution_34 = unsqueeze_273 = None
        mul_434: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_275);  sub_147 = unsqueeze_275 = None
        unsqueeze_276: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_277: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_435: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_277);  mul_434 = unsqueeze_277 = None
        unsqueeze_278: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_279: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_644: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_435, unsqueeze_279);  mul_435 = unsqueeze_279 = None
        convert_element_type_140: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_644, torch.bfloat16);  add_644 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_31: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_140);  convert_element_type_140 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_141: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        convolution_35: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_31, convert_element_type_141, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_31 = convert_element_type_141 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_660: "f32[256]" = torch.ops.aten.add.Tensor(arg179_1, 1e-05);  arg179_1 = None
        sqrt_35: "f32[256]" = torch.ops.aten.sqrt.default(add_660);  add_660 = None
        reciprocal_35: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_445: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_280: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_281: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        unsqueeze_282: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_445, -1);  mul_445 = None
        unsqueeze_283: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        sub_151: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  convolution_35 = unsqueeze_281 = None
        mul_446: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_283);  sub_151 = unsqueeze_283 = None
        unsqueeze_284: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_285: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_447: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_446, unsqueeze_285);  mul_446 = unsqueeze_285 = None
        unsqueeze_286: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg181_1, -1);  arg181_1 = None
        unsqueeze_287: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_661: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_287);  mul_447 = unsqueeze_287 = None
        convert_element_type_144: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_661, torch.bfloat16);  add_661 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_32: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_144);  convert_element_type_144 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_145: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        convolution_36: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_32, convert_element_type_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_32 = convert_element_type_145 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_677: "f32[1024]" = torch.ops.aten.add.Tensor(arg184_1, 1e-05);  arg184_1 = None
        sqrt_36: "f32[1024]" = torch.ops.aten.sqrt.default(add_677);  add_677 = None
        reciprocal_36: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_457: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_288: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_289: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        unsqueeze_290: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_457, -1);  mul_457 = None
        unsqueeze_291: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        sub_155: "f32[s77, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  convolution_36 = unsqueeze_289 = None
        mul_458: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_291);  sub_155 = unsqueeze_291 = None
        unsqueeze_292: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_293: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_459: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_458, unsqueeze_293);  mul_458 = unsqueeze_293 = None
        unsqueeze_294: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, -1);  arg186_1 = None
        unsqueeze_295: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_678: "f32[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_459, unsqueeze_295);  mul_459 = unsqueeze_295 = None
        convert_element_type_148: "bf16[s77, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_678, torch.bfloat16);  add_678 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:123 in forward, code: add_10 = layer3_3_bn3 + layer3_2_relu_2;  layer3_3_bn3 = layer3_2_relu_2 = None
        add_684: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_148, relu_30);  convert_element_type_148 = relu_30 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_33: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.relu.default(add_684);  add_684 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_149: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.bfloat16);  arg187_1 = None
        convolution_37: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_33, convert_element_type_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_149 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_700: "f32[256]" = torch.ops.aten.add.Tensor(arg189_1, 1e-05);  arg189_1 = None
        sqrt_37: "f32[256]" = torch.ops.aten.sqrt.default(add_700);  add_700 = None
        reciprocal_37: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_471: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_296: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_297: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        unsqueeze_298: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_471, -1);  mul_471 = None
        unsqueeze_299: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        sub_160: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  convolution_37 = unsqueeze_297 = None
        mul_472: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_299);  sub_160 = unsqueeze_299 = None
        unsqueeze_300: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_301: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_473: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_301);  mul_472 = unsqueeze_301 = None
        unsqueeze_302: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg191_1, -1);  arg191_1 = None
        unsqueeze_303: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_701: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_473, unsqueeze_303);  mul_473 = unsqueeze_303 = None
        convert_element_type_152: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_701, torch.bfloat16);  add_701 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_34: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_152);  convert_element_type_152 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_153: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.bfloat16);  arg192_1 = None
        convolution_38: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_34, convert_element_type_153, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_34 = convert_element_type_153 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_717: "f32[256]" = torch.ops.aten.add.Tensor(arg194_1, 1e-05);  arg194_1 = None
        sqrt_38: "f32[256]" = torch.ops.aten.sqrt.default(add_717);  add_717 = None
        reciprocal_38: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_483: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_305: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_483, -1);  mul_483 = None
        unsqueeze_307: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_164: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  convolution_38 = unsqueeze_305 = None
        mul_484: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_307);  sub_164 = unsqueeze_307 = None
        unsqueeze_308: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_309: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_485: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_484, unsqueeze_309);  mul_484 = unsqueeze_309 = None
        unsqueeze_310: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_311: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_718: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_485, unsqueeze_311);  mul_485 = unsqueeze_311 = None
        convert_element_type_156: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_718, torch.bfloat16);  add_718 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_35: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_156);  convert_element_type_156 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_157: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.bfloat16);  arg197_1 = None
        convolution_39: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_35, convert_element_type_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_35 = convert_element_type_157 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_734: "f32[1024]" = torch.ops.aten.add.Tensor(arg199_1, 1e-05);  arg199_1 = None
        sqrt_39: "f32[1024]" = torch.ops.aten.sqrt.default(add_734);  add_734 = None
        reciprocal_39: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_495: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_313: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_495, -1);  mul_495 = None
        unsqueeze_315: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_168: "f32[s77, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  convolution_39 = unsqueeze_313 = None
        mul_496: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_315);  sub_168 = unsqueeze_315 = None
        unsqueeze_316: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_317: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_497: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_317);  mul_496 = unsqueeze_317 = None
        unsqueeze_318: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg201_1, -1);  arg201_1 = None
        unsqueeze_319: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_735: "f32[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_497, unsqueeze_319);  mul_497 = unsqueeze_319 = None
        convert_element_type_160: "bf16[s77, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_735, torch.bfloat16);  add_735 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:133 in forward, code: add_11 = layer3_4_bn3 + layer3_3_relu_2;  layer3_4_bn3 = layer3_3_relu_2 = None
        add_741: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_160, relu_33);  convert_element_type_160 = relu_33 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_36: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.relu.default(add_741);  add_741 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_161: "bf16[256, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(arg202_1, torch.bfloat16);  arg202_1 = None
        convolution_40: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_36, convert_element_type_161, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_161 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_757: "f32[256]" = torch.ops.aten.add.Tensor(arg204_1, 1e-05);  arg204_1 = None
        sqrt_40: "f32[256]" = torch.ops.aten.sqrt.default(add_757);  add_757 = None
        reciprocal_40: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_509: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_509, -1);  mul_509 = None
        unsqueeze_323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_173: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  convolution_40 = unsqueeze_321 = None
        mul_510: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_323);  sub_173 = unsqueeze_323 = None
        unsqueeze_324: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_325: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_511: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_510, unsqueeze_325);  mul_510 = unsqueeze_325 = None
        unsqueeze_326: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_327: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_758: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_511, unsqueeze_327);  mul_511 = unsqueeze_327 = None
        convert_element_type_164: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_758, torch.bfloat16);  add_758 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_37: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_164);  convert_element_type_164 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_165: "bf16[256, 256, 3, 3]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.bfloat16);  arg207_1 = None
        convolution_41: "bf16[s77, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_37, convert_element_type_165, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_37 = convert_element_type_165 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_774: "f32[256]" = torch.ops.aten.add.Tensor(arg209_1, 1e-05);  arg209_1 = None
        sqrt_41: "f32[256]" = torch.ops.aten.sqrt.default(add_774);  add_774 = None
        reciprocal_41: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_521: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_329: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_521, -1);  mul_521 = None
        unsqueeze_331: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_177: "f32[s77, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
        mul_522: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_331);  sub_177 = unsqueeze_331 = None
        unsqueeze_332: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_333: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_523: "f32[s77, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_522, unsqueeze_333);  mul_522 = unsqueeze_333 = None
        unsqueeze_334: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_335: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_775: "f32[s77, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_523, unsqueeze_335);  mul_523 = unsqueeze_335 = None
        convert_element_type_168: "bf16[s77, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_775, torch.bfloat16);  add_775 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_38: "bf16[s77, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_168);  convert_element_type_168 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_169: "bf16[1024, 256, 1, 1]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.bfloat16);  arg212_1 = None
        convolution_42: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_38, convert_element_type_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_38 = convert_element_type_169 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_791: "f32[1024]" = torch.ops.aten.add.Tensor(arg214_1, 1e-05);  arg214_1 = None
        sqrt_42: "f32[1024]" = torch.ops.aten.sqrt.default(add_791);  add_791 = None
        reciprocal_42: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_533: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_337: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_533, -1);  mul_533 = None
        unsqueeze_339: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_181: "f32[s77, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
        mul_534: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_339);  sub_181 = unsqueeze_339 = None
        unsqueeze_340: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_341: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_535: "f32[s77, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_534, unsqueeze_341);  mul_534 = unsqueeze_341 = None
        unsqueeze_342: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_343: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_792: "f32[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_535, unsqueeze_343);  mul_535 = unsqueeze_343 = None
        convert_element_type_172: "bf16[s77, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_792, torch.bfloat16);  add_792 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:143 in forward, code: add_12 = layer3_5_bn3 + layer3_4_relu_2;  layer3_5_bn3 = layer3_4_relu_2 = None
        add_798: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_172, relu_36);  convert_element_type_172 = relu_36 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_39: "bf16[s77, 1024, 14, 14]" = torch.ops.aten.relu.default(add_798);  add_798 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_173: "bf16[512, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(arg217_1, torch.bfloat16);  arg217_1 = None
        convolution_43: "bf16[s77, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_39, convert_element_type_173, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_173 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_814: "f32[512]" = torch.ops.aten.add.Tensor(arg219_1, 1e-05);  arg219_1 = None
        sqrt_43: "f32[512]" = torch.ops.aten.sqrt.default(add_814);  add_814 = None
        reciprocal_43: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_547: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_345: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_547, -1);  mul_547 = None
        unsqueeze_347: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_186: "f32[s77, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
        mul_548: "f32[s77, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_347);  sub_186 = unsqueeze_347 = None
        unsqueeze_348: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_349: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_549: "f32[s77, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_548, unsqueeze_349);  mul_548 = unsqueeze_349 = None
        unsqueeze_350: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_351: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_815: "f32[s77, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_549, unsqueeze_351);  mul_549 = unsqueeze_351 = None
        convert_element_type_176: "bf16[s77, 512, 14, 14]" = torch.ops.prims.convert_element_type.default(add_815, torch.bfloat16);  add_815 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_40: "bf16[s77, 512, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_176);  convert_element_type_176 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_177: "bf16[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.bfloat16);  arg222_1 = None
        convolution_44: "bf16[s77, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_40, convert_element_type_177, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_40 = convert_element_type_177 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_831: "f32[512]" = torch.ops.aten.add.Tensor(arg224_1, 1e-05);  arg224_1 = None
        sqrt_44: "f32[512]" = torch.ops.aten.sqrt.default(add_831);  add_831 = None
        reciprocal_44: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_559: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_353: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_559, -1);  mul_559 = None
        unsqueeze_355: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_190: "f32[s77, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
        mul_560: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_355);  sub_190 = unsqueeze_355 = None
        unsqueeze_356: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_357: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_561: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_357);  mul_560 = unsqueeze_357 = None
        unsqueeze_358: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_359: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_832: "f32[s77, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_561, unsqueeze_359);  mul_561 = unsqueeze_359 = None
        convert_element_type_180: "bf16[s77, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_832, torch.bfloat16);  add_832 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_41: "bf16[s77, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_180);  convert_element_type_180 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_181: "bf16[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.bfloat16);  arg227_1 = None
        convolution_45: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_41, convert_element_type_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_41 = convert_element_type_181 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_848: "f32[2048]" = torch.ops.aten.add.Tensor(arg229_1, 1e-05);  arg229_1 = None
        sqrt_45: "f32[2048]" = torch.ops.aten.sqrt.default(add_848);  add_848 = None
        reciprocal_45: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_571: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_361: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_571, -1);  mul_571 = None
        unsqueeze_363: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_194: "f32[s77, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
        mul_572: "f32[s77, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_363);  sub_194 = unsqueeze_363 = None
        unsqueeze_364: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_365: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_573: "f32[s77, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_572, unsqueeze_365);  mul_572 = unsqueeze_365 = None
        unsqueeze_366: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_367: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_849: "f32[s77, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_367);  mul_573 = unsqueeze_367 = None
        convert_element_type_184: "bf16[s77, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(add_849, torch.bfloat16);  add_849 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_185: "bf16[2048, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.bfloat16);  arg232_1 = None
        convolution_46: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_39, convert_element_type_185, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_39 = convert_element_type_185 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_860: "f32[2048]" = torch.ops.aten.add.Tensor(arg234_1, 1e-05);  arg234_1 = None
        sqrt_46: "f32[2048]" = torch.ops.aten.sqrt.default(add_860);  add_860 = None
        reciprocal_46: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_581: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_369: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_581, -1);  mul_581 = None
        unsqueeze_371: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_197: "f32[s77, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
        mul_582: "f32[s77, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_371);  sub_197 = unsqueeze_371 = None
        unsqueeze_372: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_373: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_583: "f32[s77, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_582, unsqueeze_373);  mul_582 = unsqueeze_373 = None
        unsqueeze_374: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_375: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_861: "f32[s77, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_583, unsqueeze_375);  mul_583 = unsqueeze_375 = None
        convert_element_type_188: "bf16[s77, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(add_861, torch.bfloat16);  add_861 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:155 in forward, code: add_13 = layer4_0_bn3 + layer4_0_downsample_1;  layer4_0_bn3 = layer4_0_downsample_1 = None
        add_867: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_184, convert_element_type_188);  convert_element_type_184 = convert_element_type_188 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_42: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.relu.default(add_867);  add_867 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_189: "bf16[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.bfloat16);  arg237_1 = None
        convolution_47: "bf16[s77, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_42, convert_element_type_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_189 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_883: "f32[512]" = torch.ops.aten.add.Tensor(arg239_1, 1e-05);  arg239_1 = None
        sqrt_47: "f32[512]" = torch.ops.aten.sqrt.default(add_883);  add_883 = None
        reciprocal_47: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_595: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_377: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_595, -1);  mul_595 = None
        unsqueeze_379: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_202: "f32[s77, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
        mul_596: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_379);  sub_202 = unsqueeze_379 = None
        unsqueeze_380: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_381: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_597: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_596, unsqueeze_381);  mul_596 = unsqueeze_381 = None
        unsqueeze_382: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg241_1, -1);  arg241_1 = None
        unsqueeze_383: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_884: "f32[s77, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_597, unsqueeze_383);  mul_597 = unsqueeze_383 = None
        convert_element_type_192: "bf16[s77, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_884, torch.bfloat16);  add_884 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_43: "bf16[s77, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_192);  convert_element_type_192 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_193: "bf16[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.bfloat16);  arg242_1 = None
        convolution_48: "bf16[s77, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_43, convert_element_type_193, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_43 = convert_element_type_193 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_900: "f32[512]" = torch.ops.aten.add.Tensor(arg244_1, 1e-05);  arg244_1 = None
        sqrt_48: "f32[512]" = torch.ops.aten.sqrt.default(add_900);  add_900 = None
        reciprocal_48: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_607: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg243_1, -1);  arg243_1 = None
        unsqueeze_385: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_607, -1);  mul_607 = None
        unsqueeze_387: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_206: "f32[s77, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
        mul_608: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_387);  sub_206 = unsqueeze_387 = None
        unsqueeze_388: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_389: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_609: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_608, unsqueeze_389);  mul_608 = unsqueeze_389 = None
        unsqueeze_390: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg246_1, -1);  arg246_1 = None
        unsqueeze_391: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_901: "f32[s77, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_609, unsqueeze_391);  mul_609 = unsqueeze_391 = None
        convert_element_type_196: "bf16[s77, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_901, torch.bfloat16);  add_901 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_44: "bf16[s77, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_196);  convert_element_type_196 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_197: "bf16[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg247_1, torch.bfloat16);  arg247_1 = None
        convolution_49: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_44, convert_element_type_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_44 = convert_element_type_197 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_917: "f32[2048]" = torch.ops.aten.add.Tensor(arg249_1, 1e-05);  arg249_1 = None
        sqrt_49: "f32[2048]" = torch.ops.aten.sqrt.default(add_917);  add_917 = None
        reciprocal_49: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_619: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_393: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_619, -1);  mul_619 = None
        unsqueeze_395: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_210: "f32[s77, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
        mul_620: "f32[s77, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_395);  sub_210 = unsqueeze_395 = None
        unsqueeze_396: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_397: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_621: "f32[s77, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_620, unsqueeze_397);  mul_620 = unsqueeze_397 = None
        unsqueeze_398: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_399: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_918: "f32[s77, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_621, unsqueeze_399);  mul_621 = unsqueeze_399 = None
        convert_element_type_200: "bf16[s77, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(add_918, torch.bfloat16);  add_918 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:165 in forward, code: add_14 = layer4_1_bn3 + layer4_0_relu_2;  layer4_1_bn3 = layer4_0_relu_2 = None
        add_924: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_200, relu_42);  convert_element_type_200 = relu_42 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_45: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.relu.default(add_924);  add_924 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_201: "bf16[512, 2048, 1, 1]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.bfloat16);  arg252_1 = None
        convolution_50: "bf16[s77, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_45, convert_element_type_201, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_201 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_940: "f32[512]" = torch.ops.aten.add.Tensor(arg254_1, 1e-05);  arg254_1 = None
        sqrt_50: "f32[512]" = torch.ops.aten.sqrt.default(add_940);  add_940 = None
        reciprocal_50: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_633: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg253_1, -1);  arg253_1 = None
        unsqueeze_401: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_633, -1);  mul_633 = None
        unsqueeze_403: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_215: "f32[s77, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
        mul_634: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_403);  sub_215 = unsqueeze_403 = None
        unsqueeze_404: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_405: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_635: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_634, unsqueeze_405);  mul_634 = unsqueeze_405 = None
        unsqueeze_406: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_407: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_941: "f32[s77, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_635, unsqueeze_407);  mul_635 = unsqueeze_407 = None
        convert_element_type_204: "bf16[s77, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_941, torch.bfloat16);  add_941 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_46: "bf16[s77, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_204);  convert_element_type_204 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_205: "bf16[512, 512, 3, 3]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.bfloat16);  arg257_1 = None
        convolution_51: "bf16[s77, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_46, convert_element_type_205, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_46 = convert_element_type_205 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_957: "f32[512]" = torch.ops.aten.add.Tensor(arg259_1, 1e-05);  arg259_1 = None
        sqrt_51: "f32[512]" = torch.ops.aten.sqrt.default(add_957);  add_957 = None
        reciprocal_51: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_645: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_409: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_645, -1);  mul_645 = None
        unsqueeze_411: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_219: "f32[s77, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
        mul_646: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_411);  sub_219 = unsqueeze_411 = None
        unsqueeze_412: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_413: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_647: "f32[s77, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_646, unsqueeze_413);  mul_646 = unsqueeze_413 = None
        unsqueeze_414: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_415: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_958: "f32[s77, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_647, unsqueeze_415);  mul_647 = unsqueeze_415 = None
        convert_element_type_208: "bf16[s77, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_958, torch.bfloat16);  add_958 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_47: "bf16[s77, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_208);  convert_element_type_208 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:78 in forward, code: return super().forward(x)
        convert_element_type_209: "bf16[2048, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(arg262_1, torch.bfloat16);  arg262_1 = None
        convolution_52: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_47, convert_element_type_209, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_47 = convert_element_type_209 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:121 in forward, code: return super().forward(x)
        add_974: "f32[2048]" = torch.ops.aten.add.Tensor(arg264_1, 1e-05);  arg264_1 = None
        sqrt_52: "f32[2048]" = torch.ops.aten.sqrt.default(add_974);  add_974 = None
        reciprocal_52: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_657: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, -1);  arg263_1 = None
        unsqueeze_417: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_657, -1);  mul_657 = None
        unsqueeze_419: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_223: "f32[s77, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
        mul_658: "f32[s77, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_419);  sub_223 = unsqueeze_419 = None
        unsqueeze_420: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_421: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_659: "f32[s77, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_421);  mul_658 = unsqueeze_421 = None
        unsqueeze_422: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_423: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_975: "f32[s77, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_659, unsqueeze_423);  mul_659 = unsqueeze_423 = None
        convert_element_type_212: "bf16[s77, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(add_975, torch.bfloat16);  add_975 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:175 in forward, code: add_15 = layer4_2_bn3 + layer4_1_relu_2;  layer4_2_bn3 = layer4_1_relu_2 = None
        add_981: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_212, relu_45);  convert_element_type_212 = relu_45 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:176 in forward, code: return super().forward(x)
        relu_48: "bf16[s77, 2048, 7, 7]" = torch.ops.aten.relu.default(add_981);  add_981 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:177 in forward, code: avgpool = self.avgpool(layer4_2_relu_2);  layer4_2_relu_2 = None
        mean: "bf16[s77, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_48, [-1, -2], True);  relu_48 = None
        
         # File: <eval_with_key>.5 from /home/hice1/yyu496/.conda/envs/lib/lib/python3.10/site-packages/torchvision/models/resnet.py:284 in forward:178 in forward, code: flatten = torch.flatten(avgpool, 1);  avgpool = None
        view: "bf16[s77, 2048]" = torch.ops.aten.view.default(mean, [arg1_1, 2048]);  mean = arg1_1 = None
        
         # File: /home/hice1/yyu496/kaggle/CW/ACT6/layers.py:24 in forward, code: return super().forward(x)
        convert_element_type_213: "bf16[100]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.bfloat16);  arg268_1 = None
        convert_element_type_214: "bf16[100, 2048]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.bfloat16);  arg267_1 = None
        permute: "bf16[2048, 100]" = torch.ops.aten.permute.default(convert_element_type_214, [1, 0]);  convert_element_type_214 = None
        addmm: "bf16[s77, 100]" = torch.ops.aten.addmm.default(convert_element_type_213, view, permute);  convert_element_type_213 = view = permute = None
        return (addmm,)
        